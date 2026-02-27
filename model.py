"""
ONNX-exportable reimplementation of the Mitra Tab2D model.

Replaces non-exportable ops from the upstream AutoGluon implementation:
- einops.rearrange / einx.rearrange -> reshape/permute/unsqueeze
- einx.sum -> torch.sum
- torch.vmap(torch.bucketize) -> torch.searchsorted
- in-place masked assignment -> torch.where
- einops.pack/unpack -> torch.cat/split
- torch.utils.checkpoint.checkpoint -> direct call (eval mode)

State dict keys match upstream exactly so safetensors load without renaming.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Tab2DQuantileEmbeddingX(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(
        self,
        x_support: torch.Tensor,   # (b, s, f)
        x_query__: torch.Tensor,   # (b, n_q, f)
        padding_mask: torch.Tensor, # (b, s) bool, 1=padded
        feature_mask: torch.Tensor, # (b, f) bool, 1=padded
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = padding_mask.shape[0]
        n_s = x_support.shape[1]
        n_q = x_query__.shape[1]
        n_f = x_support.shape[2]

        # seq_len: number of valid (non-padded) observations per batch
        seq_len = (~padding_mask).float().sum(dim=1, keepdim=True)  # (b, 1)

        # Set padded observations to 9999 so they don't affect quantile calculation
        mask_expanded = padding_mask.unsqueeze(-1).expand_as(x_support)  # (b, s, f)
        x_support = torch.where(mask_expanded, torch.full_like(x_support, 9999.0), x_support)

        # Compute quantiles via sort + linear interpolation (torch.quantile is not ONNX-exportable)
        # Sort x_support along observation dim
        sorted_x, _ = torch.sort(x_support, dim=1)  # (b, s, f)

        # Quantile positions: [0.001, 0.002, ..., 0.999]
        q = torch.arange(1, 1000, dtype=torch.float, device=x_support.device) / 1000  # (999,)

        # Compute fractional indices: q * (n_s - 1)
        indices_float = q * (n_s - 1)  # (999,)
        indices_low = indices_float.long()  # (999,)
        indices_high = (indices_low + 1).clamp(max=n_s - 1)  # (999,)
        frac = (indices_float - indices_low.float()).reshape(1, 999, 1)  # (1, 999, 1)

        # Gather values at low and high indices: sorted_x is (b, s, f), gather along dim=1
        # indices need shape (b, 999, f)
        idx_low = indices_low.reshape(1, 999, 1).expand(batch_size, 999, n_f)   # (b, 999, f)
        idx_high = indices_high.reshape(1, 999, 1).expand(batch_size, 999, n_f)  # (b, 999, f)
        low_vals = torch.gather(sorted_x, 1, idx_low)    # (b, 999, f)
        high_vals = torch.gather(sorted_x, 1, idx_high)  # (b, 999, f)
        quantiles = low_vals + frac * (high_vals - low_vals)  # (b, 999, f)

        # Bucketize via broadcasting comparison (searchsorted is not ONNX-exportable)
        # quantiles: (b, 999, f) -> (b, f, 999)
        quantiles = quantiles.permute(0, 2, 1)  # (b, f, 999)

        # x_support: (b, s, f) -> (b, f, s)
        xs_t = x_support.permute(0, 2, 1)  # (b, f, s)
        xq_t = x_query__.permute(0, 2, 1)  # (b, f, n_q)

        # Count how many quantile boundaries each value exceeds
        # xs_t: (b, f, s, 1) >= quantiles: (b, f, 1, 999) -> (b, f, s, 999) -> sum -> (b, f, s)
        x_support_buck = (xs_t.unsqueeze(-1) >= quantiles.unsqueeze(-2)).sum(dim=-1).float()
        x_query_buck = (xq_t.unsqueeze(-1) >= quantiles.unsqueeze(-2)).sum(dim=-1).float()

        # Reshape back: (b, f, s) -> (b, s, f)
        x_support = x_support_buck.permute(0, 2, 1)  # (b, s, f)
        x_query__ = x_query_buck.permute(0, 2, 1)    # (b, n_q, f)

        # Normalize by valid observation count
        x_support = x_support / seq_len.unsqueeze(-1)  # (b, s, f) / (b, 1, 1)
        x_query__ = x_query__ / seq_len.unsqueeze(-1)

        # Mean (ignoring padding)
        mask_expanded = padding_mask.unsqueeze(-1).expand(batch_size, n_s, n_f)
        x_support = torch.where(mask_expanded, torch.zeros_like(x_support), x_support)
        x_support_mean = x_support.sum(dim=1, keepdim=True) / seq_len.unsqueeze(-1)  # (b, 1, f)

        x_support = x_support - x_support_mean
        x_query__ = x_query__ - x_support_mean

        # Variance (ignoring padding)
        x_support = torch.where(mask_expanded, torch.zeros_like(x_support), x_support)
        x_support_var = (x_support ** 2).sum(dim=1, keepdim=True) / seq_len.unsqueeze(-1)  # (b, 1, f)

        x_support = x_support / x_support_var.sqrt()
        x_query__ = x_query__ / x_support_var.sqrt()

        # Handle singular features (zero variance) -> set to 0
        x_support = torch.where(x_support_var == 0, torch.zeros_like(x_support), x_support)
        x_query__ = torch.where(x_support_var == 0, torch.zeros_like(x_query__), x_query__)

        return x_support, x_query__


class Tab2DEmbeddingX(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.x_embedding = nn.Linear(1, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # einx.rearrange("b s f -> b s f 1", x) -> unsqueeze
        x = x.unsqueeze(-1)  # (b, s, f, 1)
        x = self.x_embedding(x)  # (b, s, f, dim)
        return x


class Tab2DEmbeddingYClasses(nn.Module):
    def __init__(self, dim: int, n_classes: int) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.y_embedding = nn.Embedding(n_classes, dim)
        self.y_mask = nn.Embedding(1, dim)

    def forward(
        self,
        y_support: torch.Tensor,       # (b, n_s)
        padding_obs_support: torch.Tensor,  # (b, n_s) bool
        n_obs_query: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = y_support.shape[0]

        y_support = y_support.long()
        # einops.rearrange("b n -> b n 1") -> unsqueeze
        y_support = y_support.unsqueeze(-1)  # (b, n_s, 1)

        # Replace in-place: y_support[padding_obs_support] = 0
        pad_mask = padding_obs_support.unsqueeze(-1)  # (b, n_s, 1)
        y_support = torch.where(pad_mask, torch.zeros_like(y_support), y_support)

        y_support = self.y_embedding(y_support)  # (b, n_s, 1, dim)

        # Replace in-place: y_support[padding_obs_support] = 0
        pad_mask_emb = padding_obs_support.unsqueeze(-1).unsqueeze(-1)  # (b, n_s, 1, 1)
        y_support = torch.where(pad_mask_emb, torch.zeros_like(y_support), y_support)

        y_query = torch.zeros((batch_size, n_obs_query, 1), device=y_support.device, dtype=torch.long)
        y_query = self.y_mask(y_query)  # (b, n_q, 1, dim)

        return y_support, y_query


class Tab2DEmbeddingYRegression(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.y_embedding = nn.Linear(1, dim)
        self.y_mask = nn.Embedding(1, dim)

    def forward(
        self,
        y_support: torch.Tensor,       # (b, n_s)
        padding_obs_support: torch.Tensor,  # (b, n_s) bool
        n_obs_query: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = y_support.shape[0]

        y_support = y_support.float()
        # einops.rearrange("b n -> b n 1") -> unsqueeze
        y_support = y_support.unsqueeze(-1)  # (b, n_s, 1)
        y_support = self.y_embedding(y_support)  # (b, n_s, dim)
        # After Linear(1, dim) on (b, n_s, 1) we get (b, n_s, dim)
        # Need (b, n_s, 1, dim) to match upstream packing
        y_support = y_support.unsqueeze(2)  # (b, n_s, 1, dim)

        # Replace in-place: y_support[padding_obs_support] = 0
        pad_mask = padding_obs_support.unsqueeze(-1).unsqueeze(-1)  # (b, n_s, 1, 1)
        y_support = torch.where(pad_mask, torch.zeros_like(y_support), y_support)

        y_query = torch.zeros((batch_size, n_obs_query, 1), device=y_support.device, dtype=torch.long)
        y_query = self.y_mask(y_query)  # (b, n_q, 1, dim)

        return y_support, y_query


class MultiheadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.o = nn.Linear(dim, dim, bias=True)

    def forward(
        self,
        query: torch.Tensor,  # (b, t, d)
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)

        b = q.shape[0]
        t_q = q.shape[1]
        t_k = k.shape[1]

        # einops.rearrange("b t (h d) -> b h t d") -> reshape + permute
        q = q.reshape(b, t_q, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, t_k, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, t_k, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        output = F.scaled_dot_product_attention(q, k, v)

        # einops.rearrange("b h t d -> b t (h d)") -> permute + reshape
        output = output.permute(0, 2, 1, 3).reshape(b, t_q, self.dim)

        output = self.o(output)
        return output


class Layer(nn.Module):
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()

        # Block 1: row attention (across observations)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.attention1 = MultiheadAttention(dim, n_heads)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 4, bias=True)
        self.linear2 = nn.Linear(dim * 4, dim, bias=True)

        # Block 2: column attention (across features)
        self.layer_norm3 = nn.LayerNorm(dim)
        self.attention2 = MultiheadAttention(dim, n_heads)
        self.layer_norm4 = nn.LayerNorm(dim)
        self.linear3 = nn.Linear(dim, dim * 4, bias=True)
        self.linear4 = nn.Linear(dim * 4, dim, bias=True)

    def forward(
        self,
        support: torch.Tensor,  # (b, s, f+1, d)
        query__: torch.Tensor,  # (b, n_q, f+1, d)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, n_s, f1, d = support.shape
        _, n_q, _, _ = query__.shape

        # --- Block 1: Row attention (across observations per feature) ---
        support_residual = support
        query_residual = query__

        support = self.layer_norm1(support)
        query__ = self.layer_norm1(query__)

        # Reshape for row attention: (b, s, f+1, d) -> (b*(f+1), s, d)
        support_flat = support.permute(0, 2, 1, 3).reshape(b * f1, n_s, d)
        query_flat = query__.permute(0, 2, 1, 3).reshape(b * f1, n_q, d)

        support_att = self.attention1(support_flat, support_flat, support_flat)
        query_att = self.attention1(query_flat, support_flat, support_flat)

        # Reshape back: (b*(f+1), s, d) -> (b, s, f+1, d)
        support_att = support_att.reshape(b, f1, n_s, d).permute(0, 2, 1, 3)
        query_att = query_att.reshape(b, f1, n_q, d).permute(0, 2, 1, 3)

        support = support_residual + support_att
        query__ = query_residual + query_att

        # --- Block 1: FFN ---
        support_residual = support
        query_residual = query__

        support = self.layer_norm2(support)
        query__ = self.layer_norm2(query__)

        support = F.gelu(self.linear1(support))
        query__ = F.gelu(self.linear1(query__))

        support = self.linear2(support)
        query__ = self.linear2(query__)

        support = support_residual + support
        query__ = query_residual + query__

        # --- Block 2: Column attention (across features per observation) ---
        support_residual = support
        query_residual = query__

        support = self.layer_norm3(support)
        query__ = self.layer_norm3(query__)

        # Reshape for feature attention: (b, s, f+1, d) -> (b*s, f+1, d)
        support_feat = support.reshape(b * n_s, f1, d)
        query_feat = query__.reshape(b * n_q, f1, d)

        support_feat_att = self.attention2(support_feat, support_feat, support_feat)
        query_feat_att = self.attention2(query_feat, query_feat, query_feat)

        # Reshape back: (b*s, f+1, d) -> (b, s, f+1, d)
        support_feat_att = support_feat_att.reshape(b, n_s, f1, d)
        query_feat_att = query_feat_att.reshape(b, n_q, f1, d)

        support = support_residual + support_feat_att
        query__ = query_residual + query_feat_att

        # --- Block 2: FFN ---
        support_residual = support
        query_residual = query__

        support = self.layer_norm4(support)
        query__ = self.layer_norm4(query__)

        support = F.gelu(self.linear3(support))
        query__ = F.gelu(self.linear3(query__))

        support = self.linear4(support)
        query__ = self.linear4(query__)

        support = support_residual + support
        query__ = query_residual + query__

        return support, query__


class Tab2D(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_output: int,
        n_layers: int,
        n_heads: int,
        task: str,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.dim_output = dim_output
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.task = task.upper()

        self.x_quantile = Tab2DQuantileEmbeddingX(dim)
        self.x_embedding = Tab2DEmbeddingX(dim)

        if self.task == 'CLASSIFICATION':
            self.y_embedding = Tab2DEmbeddingYClasses(dim, dim_output)
        elif self.task == 'REGRESSION':
            if dim_output == 1:
                self.y_embedding = Tab2DEmbeddingYRegression(dim)
            else:
                self.y_embedding = Tab2DEmbeddingYClasses(dim, dim_output)
        else:
            raise ValueError(f'Task {task} not supported')

        self.layers = nn.ModuleList([Layer(dim, n_heads) for _ in range(n_layers)])
        self.final_layer_norm = nn.LayerNorm(dim)
        self.final_layer = nn.Linear(dim, dim_output, bias=True)

    def forward(
        self,
        x_support: torch.Tensor,           # (b, n_s, f)
        y_support: torch.Tensor,            # (b, n_s)
        x_query: torch.Tensor,              # (b, n_q, f)
        padding_features: torch.Tensor,     # (b, f) bool
        padding_obs_support: torch.Tensor,  # (b, n_s) bool
        padding_obs_query__: torch.Tensor,  # (b, n_q) bool
    ) -> torch.Tensor:
        batch_size = x_support.shape[0]
        n_obs_query = x_query.shape[1]

        # Quantile embedding
        x_support, x_query = self.x_quantile(
            x_support, x_query, padding_obs_support, padding_features
        )

        # Linear embedding
        x_support = self.x_embedding(x_support)  # (b, n_s, f, d)
        x_query = self.x_embedding(x_query)       # (b, n_q, f, d)

        # Y embedding
        y_support, y_query = self.y_embedding(
            y_support, padding_obs_support, n_obs_query
        )  # (b, n_s, 1, d), (b, n_q, 1, d)

        # Pack y and x: einops.pack((y, x), "b s * d") -> torch.cat on dim=2
        support = torch.cat([y_support, x_support], dim=2)  # (b, n_s, f+1, d)
        query = torch.cat([y_query, x_query], dim=2)        # (b, n_q, f+1, d)

        # Prepend False for the y feature in padding_features
        # einops.pack((zeros, padding_features), "b *") -> torch.cat on dim=1
        padding_features_y = torch.zeros(
            (batch_size, 1), device=padding_features.device, dtype=torch.bool
        )
        padding_features_full = torch.cat([padding_features_y, padding_features], dim=1)  # (b, f+1)

        # Transformer layers (CPU path, no flash attention)
        for layer in self.layers:
            support, query = layer(support, query)

        query = self.final_layer_norm(query)
        query = self.final_layer(query)  # (b, n_q, f+1, dim_output)

        # Unpack: split at position 1 in feature dimension
        # y_query = query[:, :, :1, :], x_query = query[:, :, 1:, :]
        y_query = query[:, :, 0, :]  # (b, n_q, dim_output)

        if self.task == 'REGRESSION' and self.dim_output == 1:
            y_query = y_query.squeeze(-1)  # (b, n_q)

        return y_query
