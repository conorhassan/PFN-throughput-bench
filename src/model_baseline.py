import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer
from torch.nn import functional as F


@dataclass
class Batch:
    xc: Tensor
    yc: Tensor
    xt: Tensor
    yt: Tensor


def build_mlp(dim_in: int, dim_hid: int, dim_out: int, depth: int) -> nn.Sequential:
    if depth < 2:
        raise ValueError("MLP depth must be >= 2")
    layers = [nn.Linear(dim_in, dim_hid), nn.SiLU()]
    for _ in range(depth - 2):
        layers.append(nn.Linear(dim_hid, dim_hid))
        layers.append(nn.SiLU())
    layers.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*layers)


class NonLinearEmbedder(nn.Module):
    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        dim_hid: int,
        dim_out: int,
        depth: int,
    ) -> None:
        super().__init__()
        self.embedder = build_mlp(dim_x + dim_y, dim_hid, dim_out, depth)

    def forward(self, batch: Batch) -> Tensor:
        x_y_ctx = torch.cat((batch.xc, batch.yc), dim=-1)
        x_0_tar = torch.cat((batch.xt, torch.zeros_like(batch.yt)), dim=-1)
        inp = torch.cat((x_y_ctx, x_0_tar), dim=1)
        return self.embedder(inp)


class ACETransformerEncoderLayer(TransformerEncoderLayer):
    def _sa_block(
        self,
        x: Tensor,
        num_ctx: int,
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x[:, :num_ctx, :],
            x[:, :num_ctx, :],
            attn_mask=None,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)

    def forward(
        self,
        x: Tensor,
        num_ctx: int,
        key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), num_ctx, key_padding_mask, is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, num_ctx, key_padding_mask, is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x


class TNPDEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        n_head: int,
        dropout: float,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ACETransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_head,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True,
                    activation=F.silu,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, batch: Batch, embeddings: Tensor) -> Tensor:
        num_ctx = batch.xc.shape[1]
        num_tar = batch.xt.shape[1]
        out = embeddings
        for layer in self.layers:
            out = layer(out, num_ctx=num_ctx)
        return out[:, -num_tar:]


class MixtureGaussianHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        num_components: int,
        dim_y: int = 1,
        std_min: float = 1e-3,
    ) -> None:
        super().__init__()
        self.num_components = num_components
        self.dim_y = dim_y
        self.std_min = std_min
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.SiLU(),
                    nn.Linear(dim_feedforward, dim_y * 3),
                )
                for _ in range(num_components)
            ]
        )

    def forward(self, batch: Batch, z_target: Tensor, reduce_ll: bool = True) -> dict:
        outputs = [head(z_target) for head in self.heads]
        stacked = torch.stack(outputs, dim=-1)  # [B, T, 3, K]
        raw_mean = stacked[:, :, 0, :]
        raw_std = stacked[:, :, 1, :]
        raw_logits = stacked[:, :, 2, :]

        std = F.softplus(raw_std) + self.std_min
        log_sigma = torch.log(std)
        log_weights = F.log_softmax(raw_logits, dim=-1)

        y = batch.yt
        if y.shape[-1] != 1:
            raise ValueError("yt must have shape [B, T, 1]")
        y = y.expand(-1, -1, self.num_components)

        log_norm = -0.5 * math.log(2.0 * math.pi)
        log_prob = -0.5 * ((y - raw_mean) / std) ** 2 - log_sigma + log_norm
        log_mix = log_weights + log_prob
        log_p = torch.logsumexp(log_mix, dim=-1)

        if reduce_ll:
            log_p = log_p.mean()

        loss = -log_p
        return {
            "loss": loss,
            "log_prob": log_p,
            "mixture_means": raw_mean,
            "mixture_log_sigma": log_sigma,
            "mixture_logits": raw_logits,
        }


class BaselineTransformerNP(nn.Module):
    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        d_model: int,
        dim_feedforward: int,
        n_head: int,
        num_layers: int,
        num_components: int,
        emb_depth: int = 2,
        emb_hidden: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        emb_hidden = emb_hidden or d_model
        self.embedder = NonLinearEmbedder(dim_x, dim_y, emb_hidden, d_model, emb_depth)
        self.encoder = TNPDEncoder(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            n_head=n_head,
            dropout=dropout,
            num_layers=num_layers,
        )
        self.head = MixtureGaussianHead(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            num_components=num_components,
            dim_y=dim_y,
        )

    def forward(self, batch: Batch, reduce_ll: bool = True) -> dict:
        embedding = self.embedder(batch)
        z_target = self.encoder(batch, embedding)
        return self.head(batch, z_target, reduce_ll=reduce_ll)
