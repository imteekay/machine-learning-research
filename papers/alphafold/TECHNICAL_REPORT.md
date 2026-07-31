# AlphaFold — Technical Report

Source: [google-deepmind/alphafold](https://github.com/google-deepmind/alphafold) (branch `main`), cross-referenced with the project's own notes in `README.md`.

## 1. Overview

AlphaFold is DeepMind's protein structure prediction system. The repo ships two related systems:

- **AlphaFold (monomer)** — the original CASP14-winning model, predicts a single chain's 3D structure from its amino acid sequence.
- **AlphaFold-Multimer** — extends the same architecture to predict complexes of multiple chains, given known stoichiometry.

Given an amino acid sequence, the model outputs 3D atomic coordinates, backbone torsion angles, and confidence estimates (pLDDT, PAE), implemented in JAX/Haiku.

## 2. Pipeline (end-to-end)

Entry point: `run_alphafold.py`, function `predict_structure()`.

1. **Genetic search / feature generation** — `alphafold/data/pipeline.py: DataPipeline.process()`
   - Runs `jackhmmer` against **UniRef90** and **MGnify**.
   - Runs `hhblits` (full DBs: BFD + UniRef30) or `jackhmmer` (reduced/small BFD) for the deep MSA.
   - Runs `hhsearch`/`hmmsearch` against **PDB70**/PDB seqres for template search.
   - Parses MSAs (Stockholm/A3M) via `alphafold/data/parsers.py`.
   - Builds a `FeatureDict`: sequence features, MSA features, and template features (`template_aatype`, `template_all_atom_positions/masks`, `template_backbone_affine_tensor`, `template_sum_probs`, `template_domain_names`, `template_mask`).
   - Result cached to `features.pkl`.
2. **Model inference** — per model in `model_runners`:
   - `model_runner.process_features()` batches/pads the feature dict.
   - `model_runner.predict()` runs the JAX model (see §3), returning `plddt`, `ranking_confidence`, and (for `_ptm`/multimer models) `predicted_aligned_error` / `max_predicted_aligned_error`.
3. **Output construction** — `protein.from_prediction()` (`alphafold/common/protein.py`) writes an unrelaxed PDB/mmCIF using per-residue pLDDT as the B-factor column.
4. **Structure relaxation** — `alphafold/relax/relax.py: AmberRelaxation` runs an AMBER force-field minimization to fix steric/bond-length violations, controlled by `--models_to_relax {all,best,none}` and `--use_gpu_relax`.
5. **Ranking** — models ranked by `ranking_confidence`; ranked outputs, confidence/PAE JSON, and per-stage timings are written to disk.

## 3. Model Architecture (`alphafold/model/`)

### 3.1 Top-level: recycling loop

`modules.py: AlphaFold` (Haiku module, "Jumper et al. 2021 Suppl. Alg. 2"):

- Wraps `AlphaFoldIteration` (= `EmbeddingsAndEvoformer` + output heads) in an `hk.while_loop` executed up to `num_recycle` times (default 3; CASP15 baseline used up to 20 with early stopping).
- Each recycling iteration feeds back **gradient-stopped** `prev_pos` (final atom positions), `prev_msa_first_row`, and `prev_pair` as extra inputs to the next iteration — this is how the network iteratively refines its own structure prediction.

### 3.2 Feature embedding + Evoformer (`EmbeddingsAndEvoformer`)

- Builds an **extra-MSA stack** (`extra_msa_stack_num_block = 4` blocks of `EvoformerIteration`) then the main **Evoformer stack** (`evoformer_num_block = 48` blocks), both via `alphafold/model/layer_stack.py` for memory-efficient block iteration.
- Per-block sub-modules (each block updates both the MSA representation and the pair/residue-pair representation):
  - `MSARowAttentionWithPairBias` — attention within a sequence, biased by the pair representation (num_head=8, dropout 0.15).
  - `MSAColumnAttention` / `MSAColumnGlobalAttention` — attention across sequences at a fixed position (num_head=8, dropout 0.0).
  - `OuterProductMean` — projects the MSA representation into an update to the pair representation (num_outer_channel=32).
  - `TriangleMultiplication` (incoming/outgoing) and `TriangleAttention` (starting/ending node) — enforce that the pairwise distance/torsion predictions respect 3D geometric consistency (triangle inequality), num_head=4, dropout 0.25.
  - `Transition` (MSA and pair) — per-position feed-forward blocks.
- **Template embedding**: `TemplatePairStack` (`num_block=2`) + `TemplateEmbedding`/`SingleTemplateEmbedding`, num_head=4, injects known homologous structures into the pair representation.
- Channel widths: `msa_channel=256`, `pair_channel=128`, `extra_msa_channel=64`, `seq_channel=384`.
- MSA/template sizing: `max_msa_clusters=512`, `max_extra_msa` (1024–5120 depending on preset), `max_templates=4`, `max_relative_feature=32`.

### 3.3 Output heads

- `DistogramHead` — 64-bin pairwise distance histogram (2.3125–21.6875 Å).
- `MaskedMsaHead` — MSA denoising auxiliary loss (BERT-style).
- `PredictedLDDTHead` — per-residue confidence (pLDDT).
- `PredictedAlignedErrorHead` — pairwise error estimate (PAE), used for multimer/interface confidence.
- `ExperimentallyResolvedHead` — predicts which atoms are resolved in the ground-truth structure.

### 3.4 Structure module (`folding.py`)

`StructureModule` ("Suppl. Alg. 20"), `num_layer = 8`:

- Each `FoldIteration` runs **Invariant Point Attention (IPA)** ("Suppl. Alg. 22"): computes scalar Q/K/V plus 3D point Q/K/V in each residue's local reference frame (via `QuatAffine`), transforms points to the global frame for geometry-aware attention, then back — this is what lets attention reason directly over 3D geometry rather than only sequence-space features.
- Backbone rigid-body frames are updated incrementally each iteration (rigid transform composition).
- Side-chain torsion angles predicted via `MultiRigidSidechain`.
- Losses: **FAPE** (Frame Aligned Point Error, via `all_atom.frame_aligned_point_error`) for both backbone and all-atom sidechain (with symmetry-aware ground-truth renaming); `supervised_chi_loss`; `structural_violation_loss` (steric/bond-length violations). FAPE uses `clamp_distance=10.0`, `loss_unit_distance=10.0`.

### 3.5 Atom representations (`all_atom.py`)

- **atom37**: fixed 37-slot per-residue atom layout (one slot per possible heavy-atom name across all amino acid types) — used for I/O and metrics.
- **atom14**: dense, per-residue-type-specific layout — used internally by the model for efficiency.
- `atom14_to_atom37` / `atom37_to_atom14` convert via precomputed gather indices.
- `atom37_to_frames` builds per-residue rigid backbone/sidechain reference frames consumed by IPA and FAPE.

## 4. Model presets (`--model_preset`)

| Preset | Description |
|---|---|
| `monomer` | Original CASP14 model, `num_ensemble=1` |
| `monomer_casp14` | Exact CASP14 config, `num_ensemble=8` (~8× slower, +0.1 GDT) |
| `monomer_ptm` | CASP14 model fine-tuned to also output pTM/PAE |
| `multimer` | AlphaFold-Multimer: 5 models × configurable seeds per model (`--num_multimer_predictions_per_model`, default 5 → 25 predictions total) |

## 5. AlphaFold-Multimer and v2.3.0 updates (`docs/technical_note_v2.3.0.md`)

- New multimer weights trained on data through 2021-09-30 (vs. 2018-04-30 originally) — ~30% more training structures, 4× more cryo-EM structures, 2× more large structures (>2000 residues).
- Training crop size increased from 384 → 640 residues; training chains per example 8 → 20.
- Self-distillation MGnify clustering threshold changed from >10 to >2 sequences/cluster.
- Max MSA sequences increased 1,152 → 2,048 for 3 of the 5 multimer models.
- CASP15 baseline used 20 seeds/model with up to 20 recycles and early stopping.
- Guidance: prefer multimer models whenever stoichiometry is known (including single chains); prefer monomer AlphaFold when stoichiometry is unknown (e.g., genome-scale screens) unless the chain is several thousand residues long.
- Not yet explored in depth in this report: `alphafold/model/modules_multimer.py` and `alphafold/model/folding_multimer.py`, which contain the multimer-specific architectural deltas (e.g., cross-chain positional encoding, multi-chain permutation handling).

## 6. Infrastructure / hardware

- **Compute**: requires a modern NVIDIA GPU for inference; AMBER relax step can run on GPU (default, faster) or CPU.
- **Reference GCP setup**: 12 vCPUs, 85GB RAM, 100GB boot disk + 3TB SSD data disk, A100 GPU.
- **`--db_preset`**: `reduced_dbs` (8 vCPUs, 8GB RAM, ~600GB disk, small BFD) vs. `full_dbs` (CASP14-matching, full BFD/UniRef30, larger disk footprint).
- **Genetic databases** (full set ≈ 2.62 TB unzipped / 556 GB compressed download):

  | DB | Unzipped | Download |
  |---|---|---|
  | BFD | 1.8 TB | 271.6 GB |
  | MGnify | 120 GB | 67 GB |
  | UniRef30 | 206 GB | 52.5 GB |
  | UniRef90 | 67 GB | 34 GB |
  | UniProt | 105 GB | 53 GB |
  | PDB mmCIF | 238 GB | 43 GB |
  | PDB70 | 56 GB | 19.5 GB |
  | PDB seqres | 0.2 GB | — |
  | Small BFD | 17 GB | 9.6 GB |
  | Model params | — | 5.3 GB (5 CASP14 + 5 multimer + ptm variants) |

## 7. Relationship to the project's existing notes (README.md)

The pre-existing `README.md` in this folder captures the conceptual story well (proteins → MSA/pair representations → Evoformer attention). This report grounds those concepts in the actual repo structure and code:

- "Pair Representation" / "distance matrix" ↔ `DistogramHead` + pair-track Evoformer updates (`OuterProductMean`, `TriangleAttention`, `TriangleMultiplication`).
- "MSA Representation" / evolutionary covariance ↔ `MSARowAttentionWithPairBias`, `MSAColumnAttention`, and the data pipeline's `jackhmmer`/`hhblits` genetic searches.
- "220 residual convolution blocks" in the README is an approximation/simplification — the actual repo uses **48 Evoformer blocks** (+4 extra-MSA blocks) of attention-based modules, not convolutions, plus an **8-layer structure module** with Invariant Point Attention.
- The README doesn't mention **recycling** (up to `num_recycle` iterations of the whole embedding+Evoformer+structure-module pipeline) or **FAPE loss** — both are central to how the model is trained and how confidence/accuracy is achieved at inference time.

## Verification

- Repo structure and file contents were fetched directly from GitHub (`gh api repos/google-deepmind/alphafold/...`, branch `main`) — no local clone exists in this environment.
- Suggested follow-up verification for a reader: clone the repo and run `python run_alphafold.py --help` to confirm current CLI flags, and open `alphafold/model/config.py` directly to confirm exact hyperparameter values, since these can change between releases.
