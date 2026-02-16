import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.hidden_size = config["lstm_hidden_dimension"]
        self.lstm_layers = config["lstm_layers"]
        self.dropout = config["dropout"]
        self.embedding_dim = config["embedding_dim"]
        self.seq_length = config["seq_length"]
        self.quantiles = config.get("quantiles", [0.1, 0.5, 0.9])

        # Variable counts
        self.time_varying_categorical_variables = config[
            "time_varying_categorical_variables"
        ]
        self.time_varying_real_variables = config["time_varying_real_variables"]
        self.static_categorical_variables = config.get(
            "static_categorical_variables", 0
        )
        self.static_real_variables = config.get("static_real_variables", 0)

        # --- 1. Static Embeddings ---
        self.static_embedding_layers = nn.ModuleList(
            [
                nn.Embedding(
                    config["static_embedding_vocab_sizes"][i], self.embedding_dim
                )
                for i in range(self.static_categorical_variables)
            ]
        )
        self.static_linear_layers = nn.ModuleList(
            [
                nn.Linear(1, self.embedding_dim)
                for _ in range(self.static_real_variables)
            ]
        )

        self.total_static_dim = self.embedding_dim * (
            self.static_categorical_variables + self.static_real_variables
        )
        self.static_projection = nn.Sequential(
            nn.Linear(self.total_static_dim, self.hidden_size), nn.ReLU()
        )
        self.static_to_h = nn.Linear(self.hidden_size, self.hidden_size)
        self.static_to_c = nn.Linear(self.hidden_size, self.hidden_size)

        # --- 2. Temporal Embeddings ---
        self.time_varying_embedding_layers = nn.ModuleList(
            [
                nn.Embedding(
                    config["time_varying_embedding_vocab_sizes"][i], self.embedding_dim
                )
                for i in range(self.time_varying_categorical_variables)
            ]
        )
        self.time_varying_linear_layers = nn.ModuleList(
            [
                nn.Linear(1, self.embedding_dim)
                for _ in range(self.time_varying_real_variables)
            ]
        )

        self.total_temporal_dim = self.embedding_dim * (
            self.time_varying_real_variables + self.time_varying_categorical_variables
        )
        self.temporal_projection = nn.Linear(self.total_temporal_dim, self.hidden_size)

        # --- 3. LSTM ---
        self.lstm = nn.LSTM(
            input_size=self.hidden_size * 2,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            dropout=self.dropout if self.lstm_layers > 1 else 0,
            batch_first=True,
        )

        # --- 4. Attention Pooling Layer ---
        self.attention_fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Tanh(),
            nn.Linear(self.hidden_size // 2, 1),
        )

        # --- 5. Output ---
        self.quantile_head = nn.Linear(self.hidden_size, len(self.quantiles))

    def get_static_embedding(self, x):
        embedding_vectors = []
        for i in range(self.static_categorical_variables):
            emb = self.static_embedding_layers[i](
                x["identifier"][:, 0, i].long().to(self.device)
            )
            embedding_vectors.append(emb)
        for j in range(self.static_real_variables):
            idx = self.static_categorical_variables + j
            emb = self.static_linear_layers[j](
                x["identifier"][:, 0, idx].view(-1, 1).float().to(self.device)
            )
            embedding_vectors.append(emb)
        return torch.cat(embedding_vectors, dim=1)

    def apply_temporal_embedding(self, x_input):
        embeddings = []
        for i in range(self.time_varying_real_variables):
            val = x_input[:, :, i].unsqueeze(-1).float()
            embeddings.append(self.time_varying_linear_layers[i](val))
        for i in range(self.time_varying_categorical_variables):
            idx = self.time_varying_real_variables + i
            val = x_input[:, :, idx].long()
            embeddings.append(self.time_varying_embedding_layers[i](val))
        return torch.cat(embeddings, dim=2)

    def forward(self, x):
        # Static Context
        static_embedding = self.get_static_embedding(x)
        static_context = self.static_projection(static_embedding)

        # Initial States
        h0 = (
            self.static_to_h(static_context).unsqueeze(0).repeat(self.lstm_layers, 1, 1)
        )
        c0 = (
            self.static_to_c(static_context).unsqueeze(0).repeat(self.lstm_layers, 1, 1)
        )

        # Temporal Context
        temporal_embeddings = self.apply_temporal_embedding(x["inputs"].to(self.device))
        temporal_features = self.temporal_projection(temporal_embeddings)

        static_repeated = static_context.unsqueeze(1).expand(
            -1, temporal_features.size(1), -1
        )
        lstm_input = torch.cat([temporal_features, static_repeated], dim=-1)

        # LSTM Pass
        lstm_output, _ = self.lstm(lstm_input, (h0, c0))

        # Pooling
        mask = x.get("mask")  # Shape: (Batch, Seq)

        # Compute attention scores
        attn_scores = self.attention_fc(lstm_output).squeeze(-1)  # (Batch, Seq)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.to(self.device) == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(1)  # (Batch, 1, Seq)

        # Weighted sum of hidden states
        pooled_output = torch.bmm(attn_weights, lstm_output).squeeze(
            1
        )  # (Batch, Hidden)

        # Quantile Output
        quantiles = self.quantile_head(pooled_output)
        return {"prediction": quantiles}
