# Monkey King Bang (MKB / 神珍) — Architecture & Implementation

> **Monkey King Bang: A Unified Scientific Multimodal Foundation Model**
> SAIS Team, Shanghai Academy of AI for Science (2026)
> [Technical report (PDF)](https://github.com/Shanghai-Academy-of-AI-For-Science/MKB/blob/main/docs/MKB.pdf) ·
> [Code](https://github.com/Shanghai-Academy-of-AI-For-Science/MKB) ·
> [Weights](https://huggingface.co/sais-org/MKB)

This is a technical read-through of MKB: what it is architecturally, how the code
is actually organized, and which design decisions are transferable. Claims below
are traced either to a source file in the upstream repo (`code/mkb/**`) or to a
section of the technical report.

---

## 1. At a glance

MKB is a **single checkpoint** that both *understands* and *generates* across six
scientific branches plus text — DNA, RNA, protein, small molecule, Earth-system
(weather), and medical image. The unifying claim is not "one model does many
tasks" (many models claim that by serializing everything into text tokens), but:

> heterogeneous scientific data gets a **native encoder** on the way in **and a
> native decoder on the way out**, with a shared autoregressive Transformer doing
> the contextualization in between.

| | |
|---|---|
| Backbone | Qwen3-VL-8B — 36 text layers, `D_LLM = 4096`, native mRoPE + deepstack vision tower retained |
| Total params | ~11 B (8 B backbone + native encoders/decoders + embedded SAM 3 branch) |
| Modalities | text, image/video, DNA, RNA, protein, molecule, weather, medical image |
| Native outputs | RNA sequence · SMILES string · 70-channel ERA5 field · segmentation mask · text |
| Training | Two-stage *modality-then-language* curriculum, bf16, DeepSpeed ZeRO-2, H200s |
| Headline results | 60.11 macro-avg over 20 bio benchmarks (vs 51.97 for the ~1 T Intern-S1-Pro); beats ECMWF HRES at day-10; 91.20 Dice on BiomedParse |
| License | Apache-2.0, **except** the embedded SAM 3 branch (Meta SAM License) |

### Capability matrix

| Modality | Understanding | Generation |
|---|:-:|:-:|
| Protein | ✅ | — (framework-supported, not activated) |
| RNA | ✅ | ✅ sequence design |
| DNA | ✅ | — |
| Molecule | ✅ | ✅ text → SMILES |
| Weather | — | ✅ 10-day global ERA5 0.25° forecast |
| Medical image | — | ✅ text-prompted segmentation mask |
| Text / image | ✅ | ✅ |

---

## 2. The gap MKB targets

Prior work splits into two families:

**Domain specialists** — ESM2 for proteins, DNABERT-2 / Nucleotide Transformer for
genomics, graph networks for molecules, Pangu/GraphCast/FuXi for weather,
MedSAM/BiomedParse for medical images. Strong within a domain, architecturally
incapable of cross-domain composition inside one model.

**Scientific generalists** — Nature Language Model, Biology-Instructions,
Intern-S1-Pro. These attach diverse inputs to a shared language backbone, but do
it through *text-like serialization*. Two failures follow:

- **Input side.** Dense atmospheric fields, molecular graphs, and biomedical
  images carry spatial/geometric/numerical structure that a BPE tokenizer
  destroys. A 721×1440×70 field is not a string.
- **Output side.** A text-centric model can *describe* a segmentation mask or a
  forecast; it cannot *decode* one from its hidden states. Even the
  trillion-parameter Intern-S1-Pro delegates to external predictors.

MKB's answer: keep the shared backbone, but give every modality a
structure-appropriate **encoder + adapter** on the input side and, where the
target is non-textual, a **native decoder** that reads the backbone's own hidden
states. The empirical payoff is visible in one number: off-the-shelf SAM 3 scores
35.40 Dice on BiomedParse; the same visual backbone conditioned on MKB's
instruction-aware hidden states scores **91.20**.

---

## 3. Architecture overview

```
                        ┌─────────────────── INPUTS ───────────────────┐
   text    image/video   DNA        RNA      protein    SMILES   ERA5 field   med image
     │         │          │          │          │          │          │           │
     │    Qwen3-VL     DNAConv-   RNAConv-    ESM2-      Suiren      Polaris    SAM3 Hiera
     │    ViT tower    Former     Former      150M        GAT       Swin enc.   (2016²)
     │         │          │          │          │          │          │           │
     │         │          └──────────┴──────────┴──────────┘          │           │
     │         │                     │                                │           │
     │         │            Perceiver resampler (K=64)          no resampler:     │
     │         │                     +                          dense 120×240     │
     │         │               MLP projector                    grid preserved    │
     │         │                     │                                │           │
     ▼         ▼                     ▼                                ▼           │
  ┌──────────────────────────────────────────────────────────────────────┐        │
  │  Unified multimodal token sequence   E_fused ∈ R^{B × L_tot × 4096}  │        │
  │  scientific features masked_scatter'd into <|{mod}_pad|> positions   │        │
  └──────────────────────────────────────────────────────────────────────┘        │
                                    │                                             │
                    ┌───────────────▼────────────────┐                            │
                    │  Shared autoregressive         │                            │
                    │  Transformer (Qwen3-VL-8B)     │                            │
                    │  36 layers · D = 4096 · mRoPE  │                            │
                    └───────────────┬────────────────┘                            │
                                    │  contextualized hidden states               │
                    ┌───────────────▼────────────────┐                            │
                    │   Target-modality dispatch     │                            │
                    │  select hidden states at the   │                            │
                    │  output positions, route to    │                            │
                    │  the matching decoder          │                            │
                    └─┬──────┬───────┬───────┬───────┘                            │
                      │      │       │       │                                    │
                  lm_head  RNA    Mol AR   Weather                        MedSeg proj ──┘
                    │     linear  decoder  Swin dec.                            │
                    ▼      ▼        ▼         ▼                                 ▼
                  text   RNA seq  SMILES   70-ch field                   SAM3 mask decoder
                                                                          → 576×576 mask
```

Three structural regimes, chosen per modality by how much of the native structure
must survive:

1. **Compress to a fixed latent budget** (DNA, RNA, protein, molecule) —
   variable-length sequence or graph → Perceiver resampler → **K = 64** tokens →
   MLP → 4096-d. Cheap, uniform, order-agnostic.
2. **Preserve the dense grid** (weather) — no resampler. 28,800 tokens go into the
   sequence so the 2-D latitude–longitude layout survives for the positional
   encoding and the field decoder.
3. **Dual-path late fusion** (medical image) — the image traverses *two* visual
   pathways in parallel (Qwen3-VL for task-aware semantics, SAM 3 Hiera for dense
   pixels) and they meet inside SAM 3's cross-attention.

---

## 4. The registry framework — the real engineering story

The most reusable part of the codebase is not any single encoder; it is the
plug-in system in `code/mkb/registry/`, which makes adding a modality a matter of
writing one package and touching **zero** model code.

### 4.1 Four abstract interfaces (`registry/base.py`)

```python
class BaseEncoder(nn.Module, ABC):
    def encode(self, inputs) -> Tuple[Tensor, Optional[Tensor]]:  # [B,K,D_enc], [B,K]
    @property def output_dim(self) -> int
    @property def num_latent_tokens(self) -> Optional[int]

class BaseProjector(nn.Module, ABC):
    def project(self, features: Tensor) -> Tensor        # [*, D_enc] → [*, D_llm]

class BaseDecoder(nn.Module, ABC):
    def decode(self, hidden_states, labels=None, **kw) -> Tensor
    def compute_loss(self, logits, labels, ignore_index=-100) -> Tensor
    @property def output_size(self) -> int

class BaseProcessor(ABC):
    def process_input(self, raw_input, **kw) -> Dict[str, Tensor]
    def build_placeholder(self, raw_input, is_output=False, **kw) -> Tuple[str, Any]
    @property def modality_name(self) -> str
```

`decode` / `compute_loss` are deliberately *not* abstract in the released
inference build — each decoder implements only the entry point it needs
(`generate_smiles`, `predict_from_hidden`, `logits_to_vocab_space`).

### 4.2 `ComponentRegistry` (`registry/registry.py`)

A singleton with four name→class maps populated by decorators:

```python
@ComponentRegistry.register_decoder("rna_lm_decoder")
class RNALMDecoder(BaseDecoder): ...
```

`build_modality_components(spec, llm_hidden_size)` instantiates encoder /
projector / decoder / processor from a `ModalitySpec`, threading
`llm_hidden_size` into the projector and decoder kwargs automatically.

### 4.3 `ModalityRouter` (`registry/modality_router.py`) — the dispatch layer

This replaces what would otherwise be a wall of `if/elif` in the model's forward.
It holds three `nn.ModuleDict`s (`encoders`, `projectors`, `decoders`) plus
non-parameter metadata (`_is_image_like`, `_aliases`, a trainability cache).

`encode_and_project(name, input_ids, attention_mask, **extra)`:

```python
actual = self.resolve(name)                 # alias → canonical
encoder = self.encoders[actual]

# Frozen encoders skip the autograd graph entirely — matters when the
# encoder is a 48-layer protein LM whose gradients would be discarded.
if has_trainable_enc:
    latent, _ = encoder(input_ids, attention_mask, **extra)
else:
    with torch.no_grad():
        latent, _ = encoder(input_ids, attention_mask, **extra)
    latent = latent.detach()

projected = self.projectors[actual](latent) if actual in self.projectors else latent

# Guard: NaN/Inf here would crash downstream bf16 CUBLAS matmuls.
if torch.isnan(projected).any() or torch.isinf(projected).any():
    projected = torch.nan_to_num(projected, nan=0.0, posinf=1e4, neginf=-1e4)

return projected.reshape(-1, projected.shape[-1])       # [B*K, D_llm]
```

`scatter_all(input_ids, inputs_embeds, bio_token_ids, **kwargs)` then loops over
every registered modality, looks for `{name}_input_ids` in kwargs, and for each
active one:

1. Collects extra kwargs by prefix (`mol_edge_index` → `edge_index`), skipping the
   standard three and anything ending in `_labels`.
2. Encodes + projects to `[total_tokens, D_llm]`.
3. Builds a boolean mask over *all* pad token IDs for that modality (canonical +
   aliases) and `masked_scatter`s the features into those positions.
4. Validates twice: global count (`n_pad == n_features`) **and** per-sample
   divisibility by K — the second check catches collator bugs (e.g. sequences
   swapped between samples) that the global count would let through silently.
5. Returns `image_like_grids: Dict[modality → grid_thw]` — a *dict*, not a list,
   so callers can reorder grids by text-appearance order before RoPE.

### 4.4 Special-token layout (`registry/token_manager.py`)

Every modality gets its own `<|{name}_start|>` / `<|{name}_end|>` /
`<|{name}_pad|>`. Two decisions here are worth copying:

**Frozen canonical IDs.** The reserved layout is hard-coded and asserted at load:

```
<|rna_start|>=151669  <|rna_end|>=151670  <|rna_pad|>=151671
<|dna_start|>=151672  ...  <|dna_pad|>=151674
<|protein_*|>=151675-151677   <|mol_*|>=151678-151680
<|contact_*|>=151681-151683   ← dormant reserved slot, kept so later IDs don't shift
<|weather_*|>=151684-151686   <|bio_seq_pad|>=151687
```

Even a single-modality run registers the *whole* layout. Filtering it by active
modalities would shift IDs and make checkpoints mutually incompatible.

**No embedding resize.** Qwen3-VL ships `embed_tokens` / `lm_head` padded to
151936 rows while `len(tokenizer)` is 151669 — 267 unused slots. All 19 bio tokens
fit there, landing on rows the base model has never looked up. Resizing is
explicitly disallowed (`_assert_fits_unused_slots` raises), because growing past
the padded size would change the `lm_head` softmax denominator and the bf16 matrix
alignment the base checkpoint expects.

**Warm-start.** New token embeddings are copied from semantically analogous
existing ones rather than randomly initialized:

| new token pattern | initialized from |
|---|---|
| `*_start` | `<\|vision_start\|>` (151652) |
| `*_end` | `<\|vision_end\|>` (151653) |
| `*_pad` | `<\|image_pad\|>` (151655) |

The reasoning: "start/end a modality segment" is structurally the same attention
problem as the vision case, and `*_pad` plays exactly the role `<|image_pad|>`
plays — a placeholder for encoder features. The `lm_head` rows get the same
treatment when weights aren't tied.

### 4.5 Auto-discovery

`Qwen3VLModel._discover_modality_packages()` walks `mkb.modalities.*` looking for
packages that export both `MODALITY_CONFIG_KEY` and `register_modality`:

```python
for config_attr, module_path in registry.items():
    mod_config = getattr(config, config_attr, None)
    if mod_config is None:
        continue                                    # modality disabled
    canonical = config_attr.replace("_config", "")
    if router.has_encoder(canonical) or router.has_decoder(canonical):
        continue                                    # already registered
    importlib.import_module(module_path).register_modality(
        self.modality_router, mod_config, llm_hidden_size
    )
```

So a new modality = a new package exporting `MODALITY_CONFIG_KEY`, `TOKEN_DEFS`,
and `register_modality(router, config, llm_hidden_size)`. Nothing in
`modeling_bio_qwen3_vl.py` changes.

### 4.6 Checkpoint-compatibility hooks

Two `_register_load_state_dict_pre_hook`s handle legacy layouts:

- `_remap_old_state_dict` — `model.{mod}_encoder.*` → `model.modality_router.encoders.{mod}.*`
  (and likewise for projectors), derived dynamically from registered names.
- `_remap_old_decoder_state_dict` — `{mod}_lm_head.*` → `model.modality_router.decoders.{mod}.head.*`.

There's also `_filter_incompatible_decoder(..., strict_mismatch=)`. For most
modalities a decoder shape mismatch drops the *entire* decoder section (never a
partial load, which would silently degrade fine-tuning). For **mol** it is strict
and raises — because a SMILES vocab drift would produce plausible-looking garbage
rather than an obvious failure.

### 4.7 Shared AR decoder blocks (`registry/decoder_blocks.py`)

`BioDecoderLayer` is a pre-norm block: causal self-attention with RoPE →
cross-attention to LLM hidden states → FFN. Two implementation notes:

- Self-attention uses `F.scaled_dot_product_attention(..., is_causal=True)` so the
  `[L_q, L_q]` mask is never materialized — relevant when `L_q` approaches 16 K.
- Cross-attention passes the KV padding mask as `[B, 1, 1, L_k]`, a strided view
  SDPA broadcasts internally rather than expanding to a dense
  `[B, heads, L_q, L_k]` tensor whose linear index would overflow INT32.

`RotaryEmbedding1D` builds its cos/sin cache **lazily as plain Python attributes**,
not `register_buffer(..., persistent=False)`. This recurs throughout the codebase
and is worth internalizing: `transformers>=5.0`'s `from_pretrained`
materialization path can leave non-persistent buffers pointing at uninitialized
memory (NaN/Inf, or finite garbage) if the module passed through the `meta`
device. The weather encoder/decoder, which *do* use real buffers, defend with
`_ensure_rope_buffers()` — checking `cos[0] == 1 and sin[0] == 0`, since an
`isfinite`-only check misses garbage that happens to be finite.

---

## 5. Modality pathways

Common contract for the four sequence/graph modalities: **variable-length encoder
features → Perceiver resampler with K = 64 learnable latent queries → 2-layer MLP
projector → 64 tokens in `R^4096`**.

```
Z^(m) = Resampler^(m)(H^(m); Q^(m)) ∈ R^{K × d_m}      Q^(m) ∈ R^{64 × d_m}
F^(m) = MLP^(m)(Z^(m))              ∈ R^{K × 4096}
```

The resampler is cross-attention from the latent queries to the encoder output,
plus an FFN; the protein variant optionally adds `num_layers − 1` self-attention
refinement blocks over the K latents (Flamingo/Perceiver-IO style), which costs
only O(K²) and helps at high compression ratios (K=64 over 2048 residues = 32×).

### 5.1 DNA & RNA — `modalities/{dna,rna}/`

Independently parameterized 1-D convolutional Transformers, architecturally
identical but with separate vocabularies and weights.

| | |
|---|---|
| Input | char-level, `{A,T,G,C,N}` (DNA) / `{A,U,G,C,N}` (RNA); T→U normalized for RNA; ≤2048 tokens |
| Width / depth | 512 / 8 blocks, 8 heads |
| Stem | 2× depthwise-separable Conv1d (kernel 7) with residual + LayerNorm — motif-level local patterns |
| Body | pre-norm Transformer blocks with 1-D RoPE on q/k |
| Adapter | Perceiver resampler K=64 → 2-layer MLP (`LayerNorm → Linear → GELU → Linear`) |

The RNA/DNA attention masking uses an **additive float mask filled with `-1e4`**
rather than a boolean mask. The comment explains why: newer PyTorch routes bool
`attn_mask` to the memory-efficient SDPA backend, which under bf16 can return NaN
for the whole output even when every query row has at least one valid key. A
finite `-1e4` keeps softmax stable.

**RNA generation decoder** (`rna/decoder.py`) is the minimal case — the backbone
already provides autoregressive contextualization, so the head is a single
bias-free `Linear(4096 → 8)` over `{pad, cls, A, U, G, C, N, sep}`. To let HF's
standard sampling loop drive generation, `logits_to_vocab_space` scatters those 8
logits into full LLM vocab space at the single-char token IDs Qwen3-VL already
has (`A`=32, `C`=34, `G`=38, `U`=52, `N`=45), mapping `<sep>` → EOS, everything
else `-inf`. DNA generation shares this decoder, so it emits `U`; callers
post-process with `.replace("U", "T")`.

### 5.2 Protein — `modalities/protein/`

Default: pretrained **ESM2-150M** (`esm2_t30_150M_UR50D`, width 640), attached via
`encoder.set_backbone()` after construction. Sequences truncated to 1024 residues.
Hidden size is validated against the backbone name — a mismatch raises with a clear
message instead of failing later inside a LayerNorm.

A self-contained fallback, `ProteinConvFormer`, mirrors `RNAConvFormer` with no ESM
dependency. Its notable extra: an optional **physicochemical feature embedding** —
a `[vocab, 6]` lookup table of normalized Kyte-Doolittle hydropathy, charge at
pH 7, side-chain volume, Grantham polarity, aromaticity, and sulfur flags,
projected by a `Linear(6 → d)` and added to the token embedding. Special and
ambiguous residues get all-zero rows, so only the 20 standard amino acids
contribute. It's a cheap inductive-bias injection for a randomly initialized
encoder before SFT.

The resampler head count adapts to non-multiple-of-64 hidden sizes (e.g. ESM2-35M's
480) by walking `n_heads` down until it divides `hidden`.

### 5.3 Molecule — `modalities/mol/`

**Encoder.** SMILES → RDKit → PyG graph. Atom and bond attributes cover atomic
identity, charge, hybridization, aromaticity, bond type, and stereochemistry. Two
complementary connectivity patterns are used simultaneously:

- `A_loc` — the sparse observed chemical bonds (functional-group structure)
- `A_full` — a fully connected graph over all atom pairs (long-range atom–atom
  exchange)

Both feed the pretrained **Suiren-ConfAvg** GAT (`graph_NN.py`, 12 layers, width
256, `NUM_ATOM_TYPE=100` covering Z ∈ [1, 99]). Node embeddings are grouped by
molecule with PyG's vectorized `to_dense_batch`, then resampled to K=64.

**Decoder** (`mol/decoder.py`) — the most interesting one, because it is
*deliberately decoupled from the LLM vocabulary*.

| | |
|---|---|
| Vocab | 226 whole-atom SMILES tokens (`MOL_VOCAB_SIZE`), IDs 0–2 = `<pad>/<cls>/<sep>` |
| Tokenization | each organic-subset atom, multi-char element, bracketed atom, bond, branch, ring closure, stereo/charge marker is one token; bracketed atoms expand into `[` / element / modifiers / `]` |
| Architecture | 6 × `BioDecoderLayer` at width 768, 12 heads |
| Conditioning | `kv_proj = Linear(4096 → 768) + LayerNorm` applied once, cached for the whole AR loop |
| Loss | shifted cross-entropy |

Because mol's vocab is not in the LLM's vocab, generation cannot ride HF's
`generate()`. `Qwen3VLForConditionalGeneration.generate_mol_smiles()` runs one
prompt forward through the LLM to get conditioning hidden states, then hands them
to `MolARDecoder.generate_smiles()`, which runs its own greedy/sampled loop in
mol-vocab space. The cross-attn KV cache is explicitly reset on every generation
mode toggle so a leftover prefill can't leak between samples.

Only `<|mol_start|>` / `<|mol_pad|>` / `<|mol_end|>` live in the LLM vocab as
placeholders.

### 5.4 Weather (Earth system) — `modalities/weather/`

The one modality that skips the latent bottleneck.

**Input.** `X_w ∈ R^{B × 1 × 70 × 721 × 1440}` — global ERA5 0.25° state.
70 channels = `z/t/u/v/q` on 13 pressure levels (50…1000 hPa) + `msl, t2m, ws10m,
u10m, v10m`. Values must arrive **already normalized** (`(physical − mean) / std`);
only the output is converted back.

**Encoder.** Polaris-style: `CubeEmbedConv` with 6×6 patches → 12 Swin blocks
(shifted-window, width 2048, 32 heads, window 20, alternating shift, AdaRMS norm,
GEGLU FFN) → `PolarisMeteoPatchMerger` (`RMSNorm → Linear → GELU → Linear`) to
4096-d.

```
(721, 1440) / 6  →  (120, 240)  →  28,800 weather tokens
```

Those 28,800 tokens go into the LLM sequence **uncompressed**, scattered into
`<|weather_pad|>` positions. The projector registered with the router is an
`IdentityWeatherProjector` — the merger already did the projection. Conditioning
comes from sinusoidal `TimestepEmbed`s over `{step, hour, doy, lead_hour}`
(`hour`/`doy` periodic), added to the hidden state, plus static geographic fields
via `const_embed`.

**Decoder.** `mlp_qwen2swin` back-projects the selected weather hidden states
4096 → 2048, adds the cached post-Swin encoder features as a skip connection,
runs 12 more Swin blocks, applies `AdaLN`, reshapes to a 2-D feature map, and
upsamples through `DoubleDeconvHead` (with the last input frame as a residual) to
the full 0.25° grid.

Encoder and decoder communicate through `encoder._step_cache` — a per-forward dict
holding `condition_embed`, `patch_embed_post_swin`, `meteo_values`, `targets`,
`lead_hours`, `input_size`, `T_in`. The decoder holds its encoder reference inside
a **1-element list** (`self._encoder_ref = [encoder]`) specifically to defeat
PyTorch's automatic submodule registration, which would otherwise duplicate every
encoder parameter under `decoders.weather.*` in the state dict.

**Rollout.** Multi-step forecasts re-feed each prediction as the next encoder input
with an updated lead-time embedding. Inference runs 6 h steps out to the requested
horizon (240 h = 10 days in the paper's evaluation).

### 5.5 Medical image — `modalities/med_seg/`

The only modality with **no encoder and no projector registered** — it scatters
nothing into the LLM embeddings. It consumes hidden states produced by Qwen3-VL's
*native* image+text path, so it adds no special tokens (`TOKEN_DEFS = {}`) and
leaves the tokenizer untouched.

```
image ─► Qwen3-VL ViT ─► LLM ─► hidden_states ──┐  (figure-text joint span)
                                                 ▼
                                        proj(4096 → 256)
                                                 │
image ─► SAM3 Hiera (2016×2016, multi-scale) ─► cross-attn ─► cls / box / mask
```

The subtlety is *which* hidden states get projected. `user_text_mask` (built by the
collator from Qwen3-VL's native image token IDs) covers the **entire user turn** —
the `<|vision_start|><|image_pad|>×N<|vision_end|>` block *and* the textual query.
After 36 self-attention layers, `image_pad`-position states encode task-aware
visual semantics and text-position states encode image-grounded language. What
SAM 3 receives as `text_embeds` is therefore a **figure–text joint representation**,
not a text condition. SAM 3 then cross-attends from its own dense Hiera features to
that representation: late fusion across two independent visual pathways.

The bridge is `LayerNorm → Linear(4096 → 512) → GELU → Linear(512 → 256)`. A
legacy single `Linear(4096 → 256)` is still selectable but compressed 16× with no
non-linearity, which bottlenecked fine-grained tasks.

`MedSegDecoder` is marked `REQUIRES_EXTRAS = True` so the router skips it on ranks
with no med-seg samples — SAM 3 cannot produce a meaningful zero-cost dummy
forward, since its image branch needs real pixels.

**Losses.** Hungarian matching over object queries (class + box + optional
mask-overlap costs), then sigmoid focal classification (α=0.25, γ=2.0), L1 box (w=5),
GIoU (w=2), sigmoid focal mask (w=2), mask Dice (w=2). Two optional auxiliaries:
an image-level semantic-mask Dice on `max_q(σ(score_q) · σ(mask_q))` against the
union of GT masks (helps multi-region targets like vessels and cells), and a
15-class meta-object cross-entropy head for coarse semantic supervision.

At inference the script fuses multiple views (full image, sliding tiles, optional
hflip, optional multi-scale) at the probability level and offers three read-out
heads:

| head | mask formation | best for |
|---|---|---|
| `argmax` | single highest-presence query | one clear target (CT / MRI / X-ray) |
| `semantic` | pixel-wise max of presence × mask-prob over queries | diffuse/multi-region (vessels, cells, OCT) |
| `union` | hard OR of all queries above `--score_thr` | several discrete instances |

Masks are predicted at 576×576 and upsampled.

---

## 6. Shared backbone & multimodal composition

### 6.1 Sequence assembly

`Qwen3VLModel.forward` runs in a fixed order:

1. Token embedding lookup.
2. `modality_router.scatter_all(...)` — all scientific modalities scattered into
   their pad positions; returns the union mask and per-modality grids.
3. **Delete consumed modality kwargs from the local dict.** This is not cosmetic:
   leaving `{mod}_input_ids`, `{mod}_edge_index`, `med_seg_pixel_values_sam3` etc.
   in `kwargs` would leak them into `language_model(**kwargs)` and pin them inside
   `GradientCheckpointingLayer` closures, wasting GPU memory across the whole
   backward.
4. Native image / video features scattered (masks OR'd with the bio mask).
5. Deepstack visual embeds assembled for `visual_pos_masks`.
6. RoPE index computation.
7. Backbone forward.

### 6.2 Modality-aware mRoPE

`get_rope_index` is the native Qwen3-VL 3-D implementation extended to recognize
per-modality start/pad tokens from `config.bio_token_ids`. Two details:

- **Grid ordering.** Grids must be consumed in the order the pad blocks appear in
  the text, not in modality-registration order. A plain `torch.cat` breaks as soon
  as a sample places `<protein>` before `<rna>` while the registry lists RNA first.
  `mkb/data/rope2d.py::_reorder_image_like_grids_by_text_pos` reorders them by
  first-occurrence position in `input_ids`.
- **Latent modalities bypass spatial merging.** Sequence/graph modalities register
  grids with `h == w == 1`; the code detects this (`is_latent`) and sets
  `merge = 1` instead of `spatial_merge_size`, so the 64 latent tokens get plain
  1-D positions while real images still get merged 2-D positions.

### 6.3 Output dispatch

`Qwen3VLForConditionalGeneration` keeps a single `_active_generation_decoder`
string, exposed through backward-compatible properties
(`_rna_generation_mode`, `_protein_generation_mode`, `_mol_generation_mode`).
Setting a mode also resets that decoder's cross-attn KV cache.

Head selection in `forward`:

```python
if self._active_generation_decoder and self._active_generation_decoder != "mol" \
        and not self.training:
    decoder = router.get_decoder(self._active_generation_decoder)
    logits = decoder.logits_to_vocab_space(hidden_sliced, vocab_size, eos_id, ...)
else:
    logits = self.lm_head(hidden_sliced)
```

So there are exactly **two output regimes**:

- **Vocab-space decoders** (RNA, protein) project into LLM vocab so HF's standard
  autoregressive loop drives generation unchanged.
- **Decoupled decoders** (mol, weather, med_seg) run outside `generate()`
  entirely — the LLM contributes only conditioning hidden states.

One further optimization: for med-seg-only training batches (no LM supervision at
all), the full-vocab `lm_head` matmul and cross-entropy are skipped
(`skip_text_logits`), saving ~0.3–1 GB of activations. This also dodges a real
bug — `cross_entropy` over zero valid tokens returns NaN and would poison the
combined loss.

---

## 7. Training curriculum

Two stages, *modality-then-language*. The motivation is stated plainly in the
paper: jointly training heterogeneous pathways in a shared backbone from scratch
gives unstable optimization and inter-modality interference.

### Stage 1 — scientific-interface alignment (backbone frozen)

| Mixture | LR | bs × ga | Trainable |
|---|---|---|---|
| Bioseq (RNA/protein/DNA) + molecule | 5e-5 | 16 × 1 | encoders + adapters; decoders where applicable |
| Weather | 1e-6 | 1 × 1 | encoder + adapter + decoder |
| MedSeg | 2e-6 | 1 × 2 | SAM 3 vision encoder + Qwen→SAM3 projection + segmentation decoder |

Each pathway learns to translate its native signal into representations the frozen
language model can consume, which preserves the pretrained language prior.

Notable per-modality wrinkles:

- **RNA + protein train jointly**, because the RNA–protein interaction dataset
  needs both encoders live in the same forward graph. To keep distributed training
  stable across RNA-only / protein-only / paired batches, the sampler keeps each
  gradient-accumulation window single-kind and synchronizes the micro-batch type
  across data-parallel ranks.
- **Weather uses truncated rollout**: unroll several consecutive 6 h steps feeding
  predictions back as input, but backprop only through the final step. Plus an EMA
  of the weather weights and patch-wise spatial dropout as augmentation.
- **MedSeg freezes the entire Qwen3-VL vision-language pathway** and the native
  SAM 3 *text* encoder; everything else in SAM 3 plus the projection is optimized.

### Stage 2 — shared-backbone consolidation

| Mixture | LR | bs × ga | Trainable |
|---|---|---|---|
| All seven modalities + scientific text + general corpora | 1e-6 | 1 × 4 | **Qwen3-VL LLM backbone + `lm_head` only** |

All Stage-1 components are loaded into a fresh base model and **frozen**. This is
the inverse of Stage 1 and the reason general capability survives: the backbone
adapts to the modality interfaces rather than the interfaces chasing a moving
backbone.

Because biological, weather, and MedSeg samples use structurally different
collators and loss paths, Stage 2 also uses single-modality micro-batches with the
type synchronized across DP ranks.

### Objective

```
L = L_text + Σ_{m ∈ M} s_m · L_m ,    s_m ∈ {0, 1},    M = {RNA, mol, weather, MedSeg}
```

Active modality losses carry equal outer weights; structured losses (weather,
MedSeg) keep the internal weights defined by their decoders. `L_text` is standard
causal LM cross-entropy over **assistant answer tokens only** — prompt tokens,
modality placeholders, and domain-output positions are all masked with
`ignore_index = -100`. Both sequence heads (RNA linear, mol AR) read hidden states
at their domain-output positions, apply a one-token causal shift, and compute CE
against domain-native labels.

The weather loss is a **latitude-weighted Charbonnier** with cosine latitude
weights (grid-cell area) and per-channel weights balancing dynamic ranges:
`α_humidity = 0.3`, `α_surface = 0.5` (t2m, 10 m winds), `α_upper = 1.0`
(upper-air + MSL).

### Infrastructure

NVIDIA H200 (141 GB), PyTorch + DeepSpeed ZeRO-2, `torchrun` over 8-GPU nodes.
Stage-1 single-domain runs span 48–248 GPUs depending on memory and sequence
length; the Stage-2 seven-modality run uses **288 GPUs (36 nodes)**. AdamW,
weight decay 0.01, cosine schedule with linear warmup, gradient clipping at global
norm 1.0, bf16 mixed precision, gradient checkpointing throughout. The language
backbone uses FlashAttention-2; the native vision tower uses SDPA for numerical
stability in the segmentation path. For large multi-node jobs the ZeRO
communication bucket sizes are reduced and communication–computation overlap is
disabled to keep medical segmentation numerically stable.

---

## 8. Data construction

### Coverage

| Domain | Sources |
|---|---|
| DNA / RNA / protein / RNA–protein | Biology-Instructions |
| Molecule | SMolInstruct, TDC, MoleHB |
| Earth system | ERA5 0.25° reanalysis (train 2002-01 → 2023-06) |
| Medical image | BiomedParse, nine imaging modalities |
| General multimodal | native image–text, OCR, grounding, instruction data inherited from Qwen3-VL |

### Unified instruction schema

The central design choice: **native modality objects are never tokenized as
prompt text**. They are stored separately from the dialogue and linked through
reserved placeholders — `<protein>`, `<rna>`, `<dna>`, `<mol>`. At data-loading
time each placeholder expands into the corresponding encoder segment, so the LM
conditions on continuous representations rather than lossy serialization.

Each sample is a three-turn dialogue: **system** fixes the task and output
contract, **human** holds the placeholders and question, **assistant** holds the
supervised answer. Only assistant tokens are supervised. For native-generation
tasks an output placeholder in the assistant turn routes those hidden states to
the target-modality decoder instead of the language head. Continuous targets are
normalized using training-set statistics only.

This shows up directly in the inference CLI:

```bash
python code/inference.py --model_path model --greedy \
  --rna "GGATGCGATCATGTCTGCACTAACACACC..." \
  --system "You are a non-coding RNA family classifier. Output only the family name." \
  --prompt $'<rna>\nWhich family does this non-coding RNA sequence belong to?'
```

### Cross-modal data (built, not found)

Ready-made instruction data is dominated by single-modality tasks. MKB builds
cross-modal supervision from authoritative databases through a **Common
Intermediate Representation (CIR)** that standardizes heterogeneous records into
typed scientific entities, relations, provenance, and evidence — decoupling
source-specific parsing from task construction.

| Type | Dataset | Modalities | Source |
|---|---|---|---|
| gen | `enzyme_substrate2product` | `<protein>` + `<mol>` → `<mol>` | UniProt + Rhea + ChEBI |
| gen | `enzyme_cofactor` | `<protein>` → `<mol>` | UniProt + ChEBI |
| reg | `davis_dti`, `kiba_dti` | `<protein>` + `<mol>` → scalar | DeepDTA |
| reg | `bindingdb_{ki,ic50,ec50,kd}` | `<protein>` + `<mol>` → scalar | BindingDB |
| cls | `rna_protein_rpi` | `<rna>` + `<protein>` → 0/1 | Biology-Instructions |

### Leakage control

Random record-level splits systematically overestimate performance on biological
data because near-duplicate sequences and scaffolds recur across the boundary. For
natively constructed datasets, **protein clusters and molecular scaffolds are
assigned at the split level**, not the record level. Exact duplicates are removed
before assignment and all splits are audited for overlap. DAVIS/KIBA keep the
official DeepDTA folds for comparability, with BindingDB cross-deduplicated
against those test pairs.

---

## 9. Results

### Biological sequence understanding (Biology-Instructions, 20 tasks)

| Task | Metric | MKB (11 B) | Biology-Instructions (8 B) | Intern-S1-Pro (~1 T) |
|---|:-:|:-:|:-:|:-:|
| DNA · EMP | MCC | **71.99** | 3.64 | 14.02 |
| DNA · PD300 | MCC | **91.17** | 58.18 | 82.65 |
| DNA · CPD | MCC | **66.35** | 44.54 | 54.60 |
| DNA · TB-H | MCC | 54.01 | 24.45 | **54.11** |
| DNA · TB-M | MCC | **65.91** | 39.91 | 60.80 |
| DNA · EA | PCC | 52.64 | 53.28 | **55.16** |
| RNA · ncRNA | Acc | **91.46** | 63.09 | 34.50 |
| RNA · APA | R² | 79.87 | 59.01 | **82.95** |
| RNA · MRL | R² | 35.54 | 47.64 | **52.41** |
| RNA · PRS | R² | 25.99 | 26.57 | **33.97** |
| RNA · Modif | AUC | **96.03** | 59.06 | 57.77 |
| RNA · CRI-On | ρ | **28.76** | −0.02 | 15.69 |
| Protein · Stability | ρ | **70.63** | 60.25 | 60.82 |
| Protein · Fluorescence | ρ | 70.12 | 2.57 | **78.14** |
| Protein · Thermostability | ρ | 46.37 | 45.07 | **59.56** |
| Protein · EC | Fmax | 68.65 | 19.79 | **72.70** |
| Protein · Solubility | Acc | 67.26 | 63.02 | **67.60** |
| Cross · AAN | MCC | 42.96 | 1.06 | **44.76** |
| Cross · RPI | MCC | **76.49** | 74.26 | 58.51 |
| Cross · EPI | MCC | −0.03 | **3.37** | −1.30 |
| **Macro average** | | **60.11** | 37.44 | 51.97 |

Best on 9/20, top-two on 17/20, and **+8.14 points over a model roughly 100×
larger**. The strength is concentrated in classification and interaction tasks;
the gaps are concentrated in scalar regression (MRL, PRS, thermostability) and
structure-dependent pairwise tasks. All three models sit near chance on EPI.

### Molecule

SMolInstruct — best or tied-best on 4/6:

| Task | Metric | MKB | Uni-Mol | LlaSMol |
|---|:-:|:-:|:-:|:-:|
| BBBP | Acc | **96.95** | 85.30 | 74.60 |
| ClinTox | Acc | 92.36 | 92.40 | **93.10** |
| HIV | Acc | **97.00** | **97.00** | 96.70 |
| SIDER | Acc | **71.00** | 70.00 | 70.70 |
| ESOL | RMSE ↓ | **0.550** | 0.819 | 1.150 |
| Lipophilicity | RMSE ↓ | 0.628 | **0.612** | 1.010 |

On the broader suites MKB trails the chemistry specialists — MoleHB normalized
average 0.8408 (vs Suiren 0.9693), ADMET average 0.7304 (vs Uni-QSAR 0.8084). This
is the breadth–specialization trade-off stated plainly: Suiren is MKB's *own*
molecule encoder, so the gap measures what the shared instruction backbone costs
on top of a fixed molecular representation.

SMILES generation: validity 89.09 / EM 22.22 / FTS 61.96, versus MolT5
(95.3 / 31.7 / 73.2) and LlaSMol (99.7 / 19.2 / 61.7) — higher EM and FTS than
LlaSMol despite lower syntactic validity.

### RNA generation

Toehold-switch design (trigger + linker → switch): BLEU 99.996, per-position
recovery 99.998%. The paper is appropriately candid that the task is largely
deterministic — the switch stem is close to a reverse-complement expansion of the
trigger plus a fixed scaffold.

### Earth-science forecasting (ERA5 0.25°, day 10, vs ECMWF HRES)

| Variable | Metric | MKB | HRES (operational NWP) |
|---|:-:|:-:|:-:|
| Z500 | RMSE ↓ | **≈680** m²/s² | ≈800 |
| Z500 | ACC ↑ | **0.64** | 0.55 |
| T2M | RMSE ↓ | **≈2.5 K** | ≈2.9 K |
| MSL | RMSE ↓ | **≈625 Pa** | ≈740 Pa |

Evaluated on a temporally disjoint 2023-07 → 2024-06 hold-out, initialized at
00/12 UTC and rolled out every 6 h to 240 h. The two systems track each other
early; MKB's advantage opens after roughly day 4 and grows — i.e. slower
medium-range skill degradation.

### Medical-image segmentation (BiomedParse, 102,855 image–prompt pairs)

Mean Dice (%):

| Modality | n | MKB | BiomedParse | MedSAM | SAM | SAM3 | DINO+MedSAM | DINO+SAM |
|---|--:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **All** | 102,855 | **91.20** | 90.73 | 83.55 | 71.29 | 35.40 | 15.37 | 15.10 |
| CT | 45,306 | **93.36** | 92.25 | 83.87 | 74.10 | 28.93 | 9.59 | 10.34 |
| MRI | 30,990 | **85.29** | 85.25 | 75.90 | 68.34 | 53.64 | 13.28 | 12.39 |
| OCT | 283 | 85.31 | **86.63** | 56.26 | 55.99 | 8.69 | 6.68 | 6.98 |
| X-ray | 13,840 | 98.02 | **98.28** | 97.75 | 81.35 | 39.96 | 37.22 | 30.63 |
| Dermoscopy | 65 | **98.08** | 97.11 | 97.35 | 88.23 | 51.47 | 81.28 | 78.29 |
| Endoscopy | 410 | **97.39** | 96.77 | 97.05 | 92.88 | 38.82 | 25.01 | 24.54 |
| Fundus | 800 | 91.33 | **91.50** | 88.06 | 57.16 | 18.58 | 3.19 | 2.73 |
| Pathology | 977 | **87.29** | 81.57 | 43.44 | 42.06 | 26.08 | 25.38 | 24.69 |
| Ultrasound | 10,184 | 90.54 | **91.03** | 89.76 | 57.47 | 5.23 | 17.12 | 22.91 |

Best on 5/9, second on the rest with small gaps. The most informative row is the
`SAM3` column: **35.40 → 91.20** using the same dense visual features plus
instruction-conditioned semantic guidance. The near-zero DINO-conditioned variants
confirm that generic visual features alone don't suffice.

### General capability retention (vs Qwen3-VL-8B-Instruct)

| Benchmark | Metric | MKB | Qwen3-VL-8B-Ins |
|---|---|:-:|:-:|
| MMLU-Pro | Acc | 73.31 | **73.36** |
| MMMU-Pro | Acc | **57.60** | 57.29 |
| AIME-2025 | Acc | **46.67** | 43.33 |
| ScreenSpot V2 | Acc | **92.30** | **92.30** |
| IMO-Answer-Bench | avg@8 | **34.94** | 34.63 |
| RefCOCO-avg | Acc@IoU≥0.5 | 88.00 | **88.01** |
| IFBench | strict prompt-level | **32.33** | **32.33** |
| OCRBench V2 ENG | score | 57.40 | **57.50** |
| OCRBench V2 CHN | score | **63.90** | 63.80 |
| SArena (Icon) | score | 71.49 | **74.83** |
| LCB V6 | pass@1 | **50.43** | 50.33 |

Match-or-beat on 7/11. Most deltas are within seed noise. The one real regression
is SArena-Icon (−3.34), attributed to limited SVG-oriented supervision in the
Stage-2 mixture. This is the strongest evidence for the freeze-then-consolidate
curriculum: adding six scientific modalities cost essentially nothing in general
capability.

---

## 10. Running it

### Install

```bash
conda create -n mkb python=3.10 -y && conda activate mkb
pip install torch==2.6.0 torchvision==0.21.0      # match your host CUDA
pip install -r requirements.txt
```

`transformers==5.0.0` is a **hard pin** — it provides both `Sam3Model` and the
`qwen3_vl` architecture used here; 5.1.x has a DeepSpeed/backward regression.
`flash-attn` is deliberately excluded from the default install (it must be
compiled against your torch/CUDA); without it the model falls back to eager
attention — identical outputs, only slower. Needs an NVIDIA GPU, ≥48 GB
recommended (≥24 GB for the 2016² SAM 3 input alone).

### Weights

```bash
hf download sais-org/MKB --local-dir ./model
```

Everything is in one `model.safetensors`: the scientific encoders/decoders (ESM-2,
Suiren GAT, the RNA/DNA ConvFormers, the Swin weather tower) **and** the fine-tuned
SAM 3 branch, whose topology and processor config live in `model/sam3/`.

### Text-style tasks

```bash
export PYTHONPATH=$PWD/code
bash run_examples.sh                    # one example per task type
GPU=1 bash run_examples.sh mol_gen      # a single example on a chosen GPU
```

Sequences pass via `--rna/--dna/--protein/--mol` and are referenced in the prompt
by `<rna>/<dna>/<protein>/<mol>`. `--system` fixes the output contract (use the
benchmark's). `--greedy` for classification/regression. `--task generation` for
RNA design, `--task mol_generation` for text→SMILES. Batch mode via
`--input_file / --output_file` with one JSON record per line.

### Weather and segmentation use dedicated scripts

```bash
# 24 h global forecast from one normalized ERA5 frame
python code/scripts/weather/infer_forecast.py \
    --checkpoint model --era5_data_path assets/era5_stats \
    --input init_frame.nc --init_time 2024-12-30T00:00:00 \
    --max_lead_hour 24 --lead_step_hours 6 --save_dir out/forecast

# text-prompted segmentation of a single image
python code/scripts/medseg/infer_med_seg_qwen3vl.py \
    --ckpt_path model --image path/to/image.png \
    --prompt "left heart ventricle in cardiac MRI" \
    --results_root out/medseg --save_vis
```

Two footguns worth naming, both documented upstream:

- The weather `--input` frame must be in **normalized space** (`(physical − mean)/std`),
  not physical units. Raw kelvin/geopotential produces wrong-scale forecasts that
  compound over the rollout.
- `--remove_channels` must match training exactly (85 → 70) — the mean/std buffers
  depend on it.
- The startup message `[weather] broken decoder RoPE buffers detected … recomputing`
  is **normal**: those tables are non-persistent recomputable constants.

### Licensing

Composite. Code and all weights **except** the SAM 3 branch are Apache-2.0. The
embedded SAM 3 medical-segmentation weights fall under Meta's **SAM License**,
which carries acceptable-use restrictions (no military/weapons/illegal use,
trade-control compliance). If you redistribute, `SAM_LICENSE.txt`, `NOTICE`, and
`THIRD_PARTY_LICENSES.md` all travel with it.

---

## 11. Design lessons

**The token budget is the real interface, and it should not be uniform.**
64 tokens per biological sequence, 28,800 for a weather field, zero for medical
images. The choice follows from what downstream consumers need: the RNA decoder
needs the backbone to have *read* the sequence, the weather decoder needs the 2-D
grid to still *be a grid*, and SAM 3 needs pixels it fetches itself. A framework
that forces one compression scheme onto every modality has already lost the
weather case.

**Adding a decoder is what separates this from adapter-tuning.** Most multimodal
work is input-side only. MKB's most striking numbers — SAM 3 at 35.40 → 91.20,
beating operational HRES — come from decoders reading the backbone's hidden
states. The backbone doesn't need to *emit* a mask; it needs to produce hidden
states from which a mask is decodable.

**Reserve your token IDs once and freeze them.** Registering the full canonical
layout even for single-modality runs (including a dormant `contact_map` slot) is
what makes any MKB checkpoint cross-loadable with any other. Filtering by active
modality would have shifted IDs and forked the checkpoint lineage permanently.

**Fit inside the existing embedding table.** Qwen3-VL's 267 padded unused rows
absorb all 19 bio tokens. Refusing to resize keeps `lm_head`'s softmax
denominator and bf16 alignment identical to the base checkpoint — the bio-token
registration is a pure no-op on original Qwen behaviour.

**Warm-start structurally analogous embeddings.** `*_start` ← `<|vision_start|>`
etc. costs nothing and hands the model a working prior for "a modality segment
begins here."

**Freeze-then-consolidate beats joint training.** Stage 1 gives each interface a
stable target; Stage 2 lets the backbone adapt to the fixed interfaces. Table 11
is the receipt — six new scientific modalities for a net-zero general-capability
change.

**Validate scatter alignment per sample, not just globally.** The global
`n_pad == n_features` check passes when two samples' sequences are swapped and K
happens to match. The per-sample `pads % K == 0` check catches it.

**Be suspicious of non-persistent buffers under `transformers>=5.0`.** The
lazy-Python-attribute RoPE caches and the `cos[0]==1 / sin[0]==0` sanity check
recur across four files because a meta-device `from_pretrained` really does leave
garbage there — and finite garbage at that, so `isfinite` alone won't catch it.

**Split by entity, not by record.** Protein-cluster and scaffold-level splits are
the difference between a real evaluation and a memorization test on biological
data.

### Limitations the authors state

- **Scalar regression is the consistent weak spot** — MRL, PRS, thermostability,
  enhancer activity, and most visibly the ADMET suite (0.7304 vs 0.8084 for
  purpose-built QSAR systems). Predicting a float through a text head is a bad fit;
  dedicated numeric regression heads are named as future work.
- **SMILES validity (89.09%) trails the specialists.** Generation isn't strongly
  constrained toward valid chemistry.
- **Enhancer–promoter interaction is at chance for every LLM-based model tested**,
  suggesting the task needs regulatory or structural priors that sequence-only
  pipelines don't carry.
- **DNA and protein generation heads exist in the framework but are not activated**
  in the reported checkpoint.
- **Weather is a single-branch pipeline** — in this release the ERA5 dataset does
  not mix with the biological data loader.

---

## 12. References

**Primary**
- SAIS Team. *Monkey King Bang: A Unified Scientific Multimodal Foundation Model.* 2026. [PDF](https://github.com/Shanghai-Academy-of-AI-For-Science/MKB/blob/main/docs/MKB.pdf)
- Code: <https://github.com/Shanghai-Academy-of-AI-For-Science/MKB> · Weights: <https://huggingface.co/sais-org/MKB>

**Components MKB builds on**
- Qwen Team. *Qwen3-VL Technical Report.* arXiv:2511.21631 — the shared backbone.
- Lin et al. *Evolutionary-scale prediction of atomic-level protein structure with a language model (ESM2).* Science, 2023 — protein encoder.
- An et al. *Suiren-1.0: A family of molecular foundation models.* 2026 — molecule graph encoder.
- Carion et al. *SAM 3: Segment Anything with Concepts.* arXiv:2511.16719 — segmentation decoder.
- Alayrac et al. *Flamingo.* NeurIPS 2022 — the cross-attention decoder pattern reused by `BioDecoderLayer`.

**Data & benchmarks**
- He et al. *Biology-Instructions.* EMNLP Findings 2025 — DNA/RNA/protein tasks and the 8 B text-token baseline.
- Yu et al. *LlaSMol / SMolInstruct.* COLM 2024.
- Huang et al. *Therapeutics Data Commons (TDC).* NeurIPS D&B 2021.
- Hersbach et al. *The ERA5 global reanalysis.* QJRMS, 2020.
- Zhao et al. *BiomedParse.* Nature Methods, 2025.

**Comparators**
- Zou et al. *Intern-S1-Pro: Scientific multimodal foundation model at trillion scale.* 2026.
- Haiden et al. *Evaluation of ECMWF forecasts (HRES).* ECMWF TR 884, 2021.
- Bi et al. *Pangu-Weather.* Nature, 2023 · Lam et al. *GraphCast.* Science, 2023.
- Ma et al. *MedSAM.* Nature Communications, 2024 · Kirillov et al. *Segment Anything.* ICCV 2023.
