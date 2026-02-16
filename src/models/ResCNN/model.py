import torch
import torch.nn as nn


class ResCNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.hidden_size = config[
            "lstm_hidden_dimension"
        ]  # Reusing config key for channel size
        self.num_blocks = config["lstm_layers"]  # Reusing layers key for depth
        self.kernel_size = 3  # Standard for time-series
        self.dropout_prob = config["dropout"]
        self.embedding_dim = config["embedding_dim"]
        self.quantiles = config.get("quantiles", [0.1, 0.5, 0.9])

        # --- Variables ---
        self.static_cat_vars = config.get("static_categorical_variables", 0)
        self.static_real_vars = config.get("static_real_variables", 0)
        self.time_cat_vars = config["time_varying_categorical_variables"]
        self.time_real_vars = config["time_varying_real_variables"]

        # --- 1. Embeddings ---
        self.static_emb = nn.ModuleList(
            [
                nn.Embedding(
                    config["static_embedding_vocab_sizes"][i], self.embedding_dim
                )
                for i in range(self.static_cat_vars)
            ]
        )
        self.time_emb = nn.ModuleList(
            [
                nn.Embedding(
                    config["time_varying_embedding_vocab_sizes"][i], self.embedding_dim
                )
                for i in range(self.time_cat_vars)
            ]
        )

        # --- 2. Input Projection ---
        # Calculate total input channels
        self.static_dim = (
            self.static_cat_vars + self.static_real_vars
        ) * self.embedding_dim
        self.temporal_dim = (
            self.time_cat_vars + self.time_real_vars
        ) * self.embedding_dim

        # We project everything to 'hidden_size' channels initially
        self.input_conv = nn.Conv1d(
            in_channels=self.temporal_dim + self.static_dim,
            out_channels=self.hidden_size,
            kernel_size=1,
        )

        # --- 3. Residual Blocks ---
        self.blocks = nn.ModuleList()
        for _ in range(self.num_blocks):
            self.blocks.append(self._make_res_block(self.hidden_size))

        # --- 4. Output Head ---
        self.head = nn.Linear(self.hidden_size, len(self.quantiles))

    def _make_res_block(self, channels):
        """Creates a standard ResNet block: Conv -> BN -> ReLU -> Dropout -> Conv -> BN"""
        return nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            ),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x):
        # 1. Process Embeddings (Same as LSTM models)
        # -------------------------------------------
        s_embs = [
            l(x["identifier"][:, 0, i].long().to(self.device))
            for i, l in enumerate(self.static_emb)
        ]
        s_reals = [
            x["identifier"][:, 0, self.static_cat_vars + i]
            .unsqueeze(-1)
            .expand(-1, self.embedding_dim)
            .float()
            .to(self.device)
            for i in range(self.static_real_vars)
        ]
        static_vec = torch.cat(s_embs + s_reals, dim=1)  # (Batch, Static_Dim)

        t_embs = [
            l(x["inputs"][:, :, self.time_real_vars + i].long().to(self.device))
            for i, l in enumerate(self.time_emb)
        ]
        t_reals = [
            x["inputs"][:, :, i]
            .unsqueeze(-1)
            .expand(-1, -1, self.embedding_dim)
            .float()
            .to(self.device)
            for i in range(self.time_real_vars)
        ]
        time_vec = torch.cat(t_reals + t_embs, dim=2)  # (Batch, Seq, Time_Dim)

        # 2. Fuse Static & Temporal
        # -------------------------
        # Repeat static features across time
        static_repeated = static_vec.unsqueeze(1).expand(-1, time_vec.size(1), -1)

        # Concatenate: (Batch, Seq, Total_Dim)
        combined = torch.cat([time_vec, static_repeated], dim=2)

        # 3. Permute for CNN
        # ------------------
        # PyTorch Conv1d expects (Batch, Channels, Seq_Len)
        # Current: (Batch, Seq_Len, Channels)
        x_cnn = combined.permute(0, 2, 1)

        # 4. Convolutional Layers
        # -----------------------
        # Initial projection
        out = self.input_conv(x_cnn)

        # Residual Stack
        for block in self.blocks:
            residual = out
            out = block(out)
            out += residual  # Skip Connection
            out = torch.relu(out)  # Final ReLU after addition

        # 5. Global Average Pooling (Masked)
        # ----------------------------------
        mask = x.get("mask").to(self.device)  # (Batch, Seq)
        if mask is not None:
            # Expand mask for channels: (Batch, Channels, Seq)
            mask = mask.unsqueeze(1)
            # Zero out padded values
            out = out * mask
            # Sum and divide by actual length
            sum_out = out.sum(dim=2)  # (Batch, Channels)
            lengths = mask.sum(dim=2)  # (Batch, 1) (broadcastable)
            pooled = sum_out / (lengths + 1e-8)
        else:
            # Standard Global Average Pooling
            pooled = out.mean(dim=2)

        # 6. Output
        return {"prediction": self.head(pooled)}
