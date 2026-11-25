"""Mini sequential, multimodal two-tower model with a search head.

  house  = HouseEncoder(tabular, geo, photo)                      shared by candidates AND history
  user   = UserTower([CLS] + house_enc(h_i) + type_i + recency_i)  self-attention, CLS pooled
  recs   = symmetric InfoNCE(user, positive house), learnable temperature, duplicate masking
  search = SearchHead([u*h, u-h, u, h, c, c*h, s]) -> 4 targets (offer, visit, share, favorite)
  total  = recs_loss + 2.0 * search_loss
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from synth_data import HIST_LEN, IMG_DIM

D = 32
N_EVENT_TYPES = 5
N_GAP = 9


def mlp(i, h, o, drop=0.1):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.LayerNorm(h), nn.Dropout(drop), nn.Linear(h, o))


class GeoTower(nn.Module):
    """'multires': embed the cell at 3 resolutions (finer => wider) + city, then MLP.
    'latlng'  : MLP on raw coordinates + city (the hand-crafted baseline)."""

    def __init__(self, n_city, n_cells, mode="multires", d=16):
        super().__init__()
        self.mode = mode
        self.city = nn.Embedding(n_city, 4)
        if mode == "multires":
            widths = (4, 8, 12)
            self.cells = nn.ModuleList(nn.Embedding(n, w) for n, w in zip(n_cells, widths))
            self.mlp = mlp(4 + sum(widths), 32, d)
        else:
            self.mlp = mlp(4 + 2, 32, d)

    def forward(self, city, cells, xy):
        parts = [self.city(city)]
        if self.mode == "multires":
            parts += [emb(cells[..., i]) for i, emb in enumerate(self.cells)]
        else:
            parts.append(xy)
        return self.mlp(torch.cat(parts, -1))


class HouseEncoder(nn.Module):
    """Tabular + geo backbone; photos enter through a zero-initialised per-dim gate alpha,
    so at step 0 the model equals the tabular model and images must earn their way in."""

    def __init__(self, world, geo_mode="multires", use_img=True):
        super().__init__()
        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt)
        self.register_buffer("city", t(world.h_city, torch.long))
        self.register_buffer("cells", t(world.h_cells, torch.long))
        self.register_buffer("xy", t(world.h_xy))
        self.register_buffer("num", t(world.h_num))
        self.register_buffer("img", t(world.h_img))
        self.geo = GeoTower(len(world.city_base), world.n_cells, geo_mode)
        self.tab = mlp(3 + 16, 64, D)
        self.use_img = use_img
        self.img_proj = mlp(IMG_DIM, 64, D)
        self.alpha = nn.Parameter(torch.zeros(D))

    def forward(self, idx):
        idx = idx.clamp(min=0)  # padding (-1) is masked downstream
        g = self.geo(self.city[idx], self.cells[idx], self.xy[idx])
        h = self.tab(torch.cat([self.num[idx], g], -1))
        if self.use_img:
            h = h + self.alpha * self.img_proj(self.img[idx])
        return h


class UserTower(nn.Module):
    """'attn': [CLS] + event tokens -> TransformerEncoder -> CLS.  'mean': masked mean."""

    def __init__(self, house_enc, mode="attn"):
        super().__init__()
        self.he, self.mode = house_enc, mode
        self.type_emb = nn.Embedding(N_EVENT_TYPES, D)
        self.gap_emb = nn.Embedding(N_GAP, D)
        self.norm = nn.LayerNorm(D)
        self.cls = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        layer = nn.TransformerEncoderLayer(D, 4, 2 * D, dropout=0.1, batch_first=True)
        self.attn = nn.TransformerEncoder(layer, 2)
        self.out = mlp(D, 2 * D, D)
        self.empty = nn.Parameter(torch.zeros(D))  # cold-start users

    def forward(self, H, T, G):
        pad = H < 0
        tok = self.norm(self.he(H) + self.type_emb(T) + self.gap_emb(G))
        if self.mode == "attn":
            x = torch.cat([self.cls.expand(len(H), -1, -1), tok], 1)
            mask = torch.cat([torch.zeros_like(pad[:, :1]), pad], 1)
            pooled = self.attn(x, src_key_padding_mask=mask)[:, 0]
        else:
            w = (~pad).float().unsqueeze(-1)
            pooled = (tok * w).sum(1) / w.sum(1).clamp(min=1)
        u = self.out(pooled)
        cold = pad.all(1, keepdim=True)
        return torch.where(cold, self.empty.expand_as(u), u)


class SearchHead(nn.Module):
    def __init__(self, n_city):
        super().__init__()
        self.city = nn.Embedding(n_city, 4)
        self.ctx = mlp(3 + 4, 64, D)  # context tower: filters + city
        self.search_only = mlp(2, 32, D)  # features that only exist at search time
        self.trunk = nn.Sequential(nn.Linear(7 * D, 128), nn.GELU(), nn.Dropout(0.1),
                                   nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 4))

    def forward(self, u, h, ctx, city, s):
        # u (B,D) h (B,K,D) ctx (B,3) s (B,K,2)
        K = h.shape[1]
        c = self.ctx(torch.cat([ctx, self.city(city)], -1)).unsqueeze(1).expand(-1, K, -1)
        u = u.unsqueeze(1).expand(-1, K, -1)
        x = torch.cat([u * h, u - h, u, h, c, c * h, self.search_only(s)], -1)
        return self.trunk(x)  # (B,K,4) logits


class MiniRecsys(nn.Module):
    def __init__(self, world, geo_mode="multires", user_mode="attn", use_img=True):
        super().__init__()
        self.house = HouseEncoder(world, geo_mode, use_img)
        self.user = UserTower(self.house, user_mode)
        self.head = SearchHead(len(world.city_base))
        self.log_inv_tau = nn.Parameter(torch.tensor(1 / 0.15).log())

    def embed_users(self, H, T, G):
        return F.normalize(self.user(H, T, G), dim=-1)

    def embed_houses(self, idx):
        return F.normalize(self.house(idx), dim=-1)

    def recs_loss(self, u, pos, w_u2h=0.8):
        """Symmetric in-batch softmax. Other rows' positives are negatives, except
        duplicates of the same house (false negatives), which are masked out."""
        h = self.embed_houses(pos)
        logits = u @ h.T * self.log_inv_tau.exp().clamp(max=100)
        dup = (pos[:, None] == pos[None, :]) & ~torch.eye(len(pos), dtype=torch.bool, device=pos.device)
        logits = logits.masked_fill(dup, -1e4)
        y = torch.arange(len(pos), device=pos.device)
        return w_u2h * F.cross_entropy(logits, y) + (1 - w_u2h) * F.cross_entropy(logits.T, y)

    def search_logits(self, u, cand, ctx, city, s):
        return self.head(u, self.embed_houses(cand), ctx, city, s)


SERVE_WEIGHTS = torch.tensor([0.3, 0.3, 0.2, 0.2])  # offer, visit, share, favorite


def serve_score(logits, pos_weight):
    """Blend calibrated per-target probabilities into one score in [0, 1].
    Subtracting log(pos_weight) undoes the class-balancing shift of the BCE logits."""
    return (torch.sigmoid(logits - pos_weight.log()) * SERVE_WEIGHTS).sum(-1)
