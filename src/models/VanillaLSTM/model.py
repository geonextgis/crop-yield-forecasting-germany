import torch
import torch.nn as nn


class VanillaLSTM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.hidden_size = config["lstm_hidden_dimension"]
        self.lstm_layers = config["lstm_layers"]
        self.embedding_dim = config["embedding_dim"]
        self.quantiles = config.get("quantiles", [0.1, 0.5, 0.9])

        # Variable counts
        self.static_cat_vars = config.get("static_categorical_variables", 0)
        self.static_real_vars = config.get("static_real_variables", 0)
        self.time_cat_vars = config["time_varying_categorical_variables"]
        self.time_real_vars = config["time_varying_real_variables"]

        # Embeddings
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

        # Projections
        self.static_in_dim = (
            self.static_cat_vars + self.static_real_vars
        ) * self.embedding_dim
        self.time_in_dim = (
            self.time_cat_vars + self.time_real_vars
        ) * self.embedding_dim

        # Simple Linear projection to mix features before LSTM
        self.input_proj = nn.Linear(
            self.time_in_dim + self.static_in_dim, self.hidden_size
        )

        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=config["dropout"] if self.lstm_layers > 1 else 0,
        )

        self.head = nn.Linear(self.hidden_size, len(self.quantiles))

    def forward(self, x):
        # 1. Process Static
        s_embs = [
            l(x["identifier"][:, 0, i].long().to(self.device))
            for i, l in enumerate(self.static_emb)
        ]
        # Map real vars to embedding dim
        s_reals = [
            x["identifier"][:, 0, self.static_cat_vars + i]
            .unsqueeze(-1)
            .expand(-1, self.embedding_dim)
            .float()
            .to(self.device)
            for i in range(self.static_real_vars)
        ]
        static_vec = torch.cat(s_embs + s_reals, dim=1)  # (Batch, Static_Dim)

        # 2. Process Temporal
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

        # 3. Concatenate Static to every time step
        static_repeated = static_vec.unsqueeze(1).expand(-1, time_vec.size(1), -1)
        combined = torch.cat([time_vec, static_repeated], dim=2)

        # 4. LSTM
        x_in = self.input_proj(combined)
        out, _ = self.lstm(x_in)

        # 5. Output
        return {"prediction": self.head(out[:, -1, :])}
