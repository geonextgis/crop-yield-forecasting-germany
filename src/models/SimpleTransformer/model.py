import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTransformer(nn.Module):
    """
    - Pure Attention (No LSTM)
    - Standard Positional Encoding
    """

    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.hidden_size = config[
            "lstm_hidden_dimension"
        ]  # Reusing config name for consistency
        self.dim_feedforward = self.hidden_size * 4
        self.embedding_dim = config["embedding_dim"]
        self.quantiles = config.get("quantiles", [0.1, 0.5, 0.9])

        # ... [Variable definitions same as VanillaLSTM above] ...
        self.static_cat_vars = config.get("static_categorical_variables", 0)
        self.static_real_vars = config.get("static_real_variables", 0)
        self.time_cat_vars = config["time_varying_categorical_variables"]
        self.time_real_vars = config["time_varying_real_variables"]

        # Embeddings (Same as above)
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

        # Input Projection
        self.static_in_dim = (
            self.static_cat_vars + self.static_real_vars
        ) * self.embedding_dim
        self.time_in_dim = (
            self.time_cat_vars + self.time_real_vars
        ) * self.embedding_dim
        self.input_proj = nn.Linear(
            self.time_in_dim + self.static_in_dim, self.hidden_size
        )

        # Positional Encoding
        self.pos_encoder = nn.Parameter(
            torch.randn(1, config["seq_length"], self.hidden_size)
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=config["attn_heads"],
            dim_feedforward=self.dim_feedforward,
            dropout=config["dropout"],
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config["lstm_layers"]
        )  # Use layer count as depth

        self.head = nn.Linear(self.hidden_size, len(self.quantiles))

    def forward(self, x):
        # 1. Embed Features (Same logic as VanillaLSTM)
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
        static_vec = torch.cat(s_embs + s_reals, dim=1)

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
        time_vec = torch.cat(t_reals + t_embs, dim=2)

        # 2. Fuse & Project
        static_repeated = static_vec.unsqueeze(1).expand(-1, time_vec.size(1), -1)
        combined = torch.cat([time_vec, static_repeated], dim=2)
        x_in = self.input_proj(combined)

        # 3. Add Positional Encoding
        x_in = x_in + self.pos_encoder[:, : x_in.size(1), :].to(self.device)

        # 4. Transformer Pass
        # Masking padding if needed (optional based on your data)
        mask = x.get("mask")
        src_key_padding_mask = (mask == 0) if mask is not None else None

        out = self.transformer(x_in, src_key_padding_mask=src_key_padding_mask)

        # 5. Pooling (Global Average for Transformers usually works better than last step)
        out_pooled = out.mean(dim=1)

        return {"prediction": self.head(out_pooled)}
