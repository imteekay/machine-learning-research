# 04-experiments

One folder per experiment. Each holds the pre-registered write-up (`README.md`), the code notes (`CODE.md`), and the runnable package (`src/`, `scripts/`, `tests/`, `results/`, `pyproject.toml`).

| ID | Question | Status | Code package |
|---|---|---|---|
| [E0.2](./E0.2-tf-knockout-targets/) | Do knockouts move their known targets? | running | `perturbseq_mini` |
| [E0.4](./E0.4-minimal-virtual-cell/) | Can a minimal virtual cell predict unseen knockouts? | done | `vcell` |
| [E1.1](./E1.1-integration-without-erasing-disease/) | Can integration remove batch without erasing disease signal? | done (minimal) | `scpoli_mini` |
| [E2.1](./E2.1-mini-scgpt-vs-pca/) | Does a tiny scGPT beat HVG + PCA for annotation? | done (mini) | `scgpt_mini` |
| [E2.3a](./E2.3a-glyph-mini-sae-concept-discovery/) | Does a Glyph-style pipeline recover known truth? | done | `glyph_mini` |
| [E3.1](./E3.1-atac-regulatory-info/) | Is the ATAC information regulatory? | done | `get_mini` |
| [E4.5](./E4.5-scgen-unseen-cell-type/) | Does latent arithmetic predict an unseen cell type's response? | done | `scgen_mini` |
| [E5.1](./E5.1-niche-context/) | Does niche context add information beyond cell type? | done (synthetic) | `nicheformer_mini` |
| [E6.2](./E6.2-histology-to-expression/) | Can histology carry the virtual cell to patients? | done (synthetic) | `omiclip_mini` |
| [E0.1](./E0.1-count-likelihoods/) | Which likelihood fits single-cell counts? | done | `scvi_mini` |
| [E1.3](./E1.3-donor-reidentification/) | How identifiable are donors in your training data? | done | `privacy_mini` |
| [E2.4](./E2.4-cross-species-transfer/) | Do protein-language gene tokens transfer cell identity from mouse to human? | planned | `uce_mini` |
| [E3.2](./E3.2-rna-atac-alignment/) | Can paired cells be matched across modalities? | planned | `scclip_mini` |
| [E4.3](./E4.3-distributional-drug-response/) | Do distributional models capture responders vs non-responders to drugs? | planned | `flowpert_mini` |
| [E5.3](./E5.3-cross-tissue-niches/) | Does a niche model generalize to unseen tissues? | planned | `novae_mini` |
| [E6.3](./E6.3-active-learning-loop/) | Does closing the loop make discovery cheaper? | planned | `loop_mini` |

Cross-experiment learnings: [`LEARNINGS.md`](./LEARNINGS.md).

Other: [`_template.md`](./_template.md) (copy into a new `E<id>-<slug>/README.md`), [`_paper-selection/`](./_paper-selection/).

Run an experiment: `cd E<id>-<slug> && uv sync && uv run pytest -q` (see its `CODE.md`).
