"""Embed every gene's canonical protein with ESM2-8M (mean-pooled, frozen). One 320-d vector per gene symbol.

    uv run python -I scripts/embed_proteins.py

Writes data/esm2_8m_{human,mouse}.npz (symbols + float16 embeddings). The weights come from the
public facebook/esm2_t6_8M_UR50D checkpoint on Hugging Face.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from uce_mini.data import load_uniprot

ROOT = Path(__file__).resolve().parents[1]
MAXLEN = 1022


@torch.no_grad()
def embed(seqs: list[str], tok, model, batch_tokens: int = 12000) -> np.ndarray:
    order = np.argsort([len(s) for s in seqs])
    out = np.zeros((len(seqs), model.config.hidden_size), np.float32)
    i = 0
    while i < len(order):
        L = min(len(seqs[order[i]]), MAXLEN) + 2
        j = i
        while j < len(order) and (j - i + 1) * (min(len(seqs[order[j]]), MAXLEN) + 2) <= batch_tokens:
            j += 1
        j = max(j, i + 1)
        idx = order[i:j]
        enc = tok([seqs[k][:MAXLEN] for k in idx], return_tensors="pt", padding=True)
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].clone()
        m[:, 0] = 0                                   # drop <cls>
        m[torch.arange(len(idx)), enc["attention_mask"].sum(1) - 1] = 0   # drop <eos>
        out[idx] = ((h * m[..., None]).sum(1) / m.sum(1, keepdim=True)).numpy()
        i = j
    return out


def main() -> None:
    tok = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D").eval()
    torch.set_num_threads(8)
    for name in ("human", "mouse"):
        prot = load_uniprot(ROOT / "data" / f"uniprot_{name}.tsv")
        genes = list(prot)
        t0 = time.time()
        E = embed([prot[g] for g in genes], tok, model)
        np.savez_compressed(ROOT / "data" / f"esm2_8m_{name}.npz", genes=np.array(genes), emb=E.astype(np.float16))
        print(name, E.shape, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
