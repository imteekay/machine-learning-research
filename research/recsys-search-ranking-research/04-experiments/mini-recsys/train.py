"""Data prep, joint training (recs InfoNCE + search head) and evaluation of one variant."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from evaluate import information_abundance, retrieval_metrics, session_ndcg
from models import MiniRecsys, serve_score
from synth_data import N_CAND, RETR_WINDOW, TRAIN_END, VAL_START, World, histories, popularity


def prepare(world: World) -> dict:
    H, T, G = histories(world, world.s_user, world.s_time)
    t_rep = np.repeat(world.s_time, N_CAND).reshape(-1, N_CAND)
    pop = popularity(world, world.s_cand, t_rep)
    S = np.stack([world.h_slots[world.s_cand], pop], -1).astype(np.float32)
    S_norm = np.stack([(S[..., 0] - 3) / 1.7, np.log1p(S[..., 1]) - 1.5], -1).astype(np.float32)
    # one positive per session for the recs loss: the candidate with the strongest intent
    strength = (world.s_labels * np.array([4, 3, 2, 1])).max(-1)
    pos = np.where(strength.max(1) > 0, world.s_cand[np.arange(len(H)), strength.argmax(1)], -1)

    # retrieval queries (online-style): for each user active in the val window, the query
    # time is their first val session; history = all events before it; positives = houses
    # with high-intent events in the next RETR_WINDOW days. Warm users only (>=1 past event);
    # cold-start users are counted but not scored (no history to embed).
    val_s = np.where(world.s_time >= VAL_START)[0]
    q_users, first = np.unique(world.s_user[val_s], return_index=True)
    q_time = world.s_time[val_s][first]
    qH, qT, qG = histories(world, q_users, q_time)
    hi = world.e_type >= 1
    q_pos = []
    for u, t in zip(q_users, q_time):
        m = hi & (world.e_user == u) & (world.e_time >= t) & (world.e_time < t + RETR_WINDOW)
        q_pos.append(np.unique(world.e_house[m]))
    keep = (qH >= 0).any(1) & np.array([len(x) > 0 for x in q_pos])
    cold_frac = float(1 - (qH >= 0).any(1).mean())
    return dict(H=H, T=T, G=G, S=S, S_norm=S_norm, pos=pos,
                train_idx=np.where(world.s_time < TRAIN_END)[0], val_idx=val_s,
                q_users=q_users[keep], qH=qH[keep], qT=qT[keep], qG=qG[keep],
                q_pos=[x for x, k in zip(q_pos, keep) if k],
                q_city=world.u_city[q_users[keep]], q_time=q_time[keep], cold_frac=cold_frac)


def batches(idx, city, bs, city_grouped, rng, p_same_city=0.8):
    """City-grouped sampler: 80% of batches come from one city -> in-batch negatives are
    same-city houses (hard). Otherwise batches mix the whole country (easy negatives)."""
    idx = rng.permutation(idx)
    if not city_grouped:
        return [idx[i:i + bs] for i in range(0, len(idx), bs)]
    pools = {c: list(idx[city[idx] == c]) for c in np.unique(city[idx])}
    mixed = list(idx)
    out = []
    for _ in range(len(idx) // bs):
        if rng.random() < p_same_city:
            c = rng.choice(list(pools))
            p = pools[c]
            if len(p) < bs:
                pools[c] = p = list(rng.permutation(idx[city[idx] == c]))
            out.append(np.array([p.pop() for _ in range(bs)]))
        else:
            out.append(rng.choice(mixed, bs, replace=False))
    return out


def train_model(world, prep, geo_mode="multires", user_mode="attn", city_grouped=True,
                use_img=True, epochs=20, bs=256, lr=3e-3, rank_weight=2.0, seed=0, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MiniRecsys(world, geo_mode, user_mode, use_img)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    tr = prep["train_idx"]
    p = world.s_labels[tr].mean((0, 1))
    pos_weight = torch.tensor((1 - p) / p, dtype=torch.float32)
    t = lambda a, dt=torch.long: torch.as_tensor(a, dtype=dt)
    for ep in range(epochs):
        model.train()
        tot = rl = sl = 0.0
        bl = batches(tr, world.s_city, bs, city_grouped, rng)
        for b in bl:
            u = model.embed_users(t(prep["H"][b]), t(prep["T"][b]), t(prep["G"][b]))
            has = prep["pos"][b] >= 0
            l_rec = model.recs_loss(u[t(has, torch.bool)], t(prep["pos"][b][has]))
            logits = model.search_logits(u, t(world.s_cand[b]), t(world.s_ctx[b], torch.float32),
                                         t(world.s_city[b]), t(prep["S_norm"][b], torch.float32))
            l_rank = F.binary_cross_entropy_with_logits(
                logits, t(world.s_labels[b], torch.float32), pos_weight=pos_weight)
            loss = l_rec + rank_weight * l_rank
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item(); rl += l_rec.item(); sl += l_rank.item()
        n = len(bl)
        log(f"  epoch {ep + 1}/{epochs} loss={tot / n:.4f} recs={rl / n:.4f} search={sl / n:.4f} "
            f"tau={1 / model.log_inv_tau.exp().item():.3f} |alpha|={model.house.alpha.abs().mean().item():.3f}")
    model.pos_weight = pos_weight
    return model


@torch.no_grad()
def eval_model(world, prep, model) -> dict:
    model.eval()
    t = lambda a, dt=torch.long: torch.as_tensor(a, dtype=dt)
    n_h = len(world.h_city)
    he = model.embed_houses(torch.arange(n_h)).numpy()
    ue = model.embed_users(t(prep["qH"]), t(prep["qT"]), t(prep["qG"])).numpy()
    sc = ue @ he.T
    out = {"retrieval_global": retrieval_metrics(sc, prep["q_pos"])}
    sc_city = np.where(world.h_city[None] == prep["q_city"][:, None], sc, -np.inf)
    out["retrieval_city"] = retrieval_metrics(sc_city, prep["q_pos"])
    out["info_abundance"] = information_abundance(he)
    out["alpha_mean_abs"] = float(model.house.alpha.abs().mean())

    va = prep["val_idx"]
    u = model.embed_users(t(prep["H"][va]), t(prep["T"][va]), t(prep["G"][va]))
    cand = t(world.s_cand[va])
    dot = (u.unsqueeze(1) * model.embed_houses(cand)).sum(-1).numpy()
    logits = model.search_logits(u, cand, t(world.s_ctx[va], torch.float32), t(world.s_city[va]),
                                 t(prep["S_norm"][va], torch.float32))
    head = serve_score(logits, model.pos_weight).numpy()
    y = world.s_labels[va]
    out["search"] = {name: {f"ndcg@5_{tg}": session_ndcg(s, y[..., i])
                            for i, tg in enumerate(["offer", "visit"])}
                     for name, s in [("dot_product", dot), ("search_head", head)]}
    out["_head_scores"] = head
    return out
