# Sybil Technical Report

An engineering reference for the Sybil lung cancer risk model codebase, combining the clinical/paper context in `knowledge-base.md` with a code-anchored walkthrough of the `sybil/` package.

Every non-obvious claim below is cited as `file:line`. Claims that come from the paper notes rather than the code are explicitly attributed to `knowledge-base.md` — the repository itself publishes no performance numbers.

**Reading order for a new engineer:** §1 → §3.7 (shape trace) → §2 (data contract) → §4 (hyperparameters) → §7 (training) → §12 (gotchas).

---

## 1. Overview

Sybil predicts an individual's **future lung cancer risk from a single low-dose chest CT (LDCT)** — the standard imaging modality in lung cancer screening. It is not a nodule detector: it is trained to predict cancer diagnosis in the 1–6 years following a scan, including on scans with no visible suspicious findings at the time.

- **Output**: six calibrated probabilities — P(diagnosis within year 1) … P(within year 6). Six, not five: `--max_followup` defaults to `6` (`sybil/parsing.py:228`) and the regression test expects six scores (`tests/regression_test.py:122-129`). The `max_followup=5` in `tests/test_create_sybilnet.py:12` is a throwaway fake arg, and `knowledge-base.md:46` ("1-5 years") is inconsistent with the rest of that document.
- **Training data**: NLST LDCTs. A scan is positive if the patient was biopsy-confirmed with lung cancer within 6 years, regardless of whether a nodule was visible on that exam (`knowledge-base.md:7`).
- **Auxiliary supervision**: for patients who developed cancer within 1 year, two fellowship-trained thoracic radiologists annotated suspicious lesions as bounding boxes via MD.ai, normalized to a 512×512 frame (`README.md:59-79`). These supervise the attention maps through a loss term only (§7) — there is no annotation head, and they are off by default (`--use_annotations`, `sybil/parsing.py:277-282`).
- **External validation**: MGH and Chang Gung Memorial Hospital (CGMH, Taiwan). At test time Sybil needs only the LDCT — no clinical data, no annotations (`knowledge-base.md:20`).
- **Headline results** (per `knowledge-base.md:37`): AUC ≈ 0.86–0.94 at 1 year, ≈ 0.74–0.81 at 6 years across validation cohorts.
- **Key limitation** (per `knowledge-base.md:33`): NLST scans are from 2002–2004 and the cohort is 92% White.
- **Input orientation contract**: axial LDCT, first frame at the abdomen, last frame at the clavicles. DICOM is auto-sorted; **PNG input must be pre-ordered by the caller** (`README.md:52-56`).

### Public API surface

```python
from sybil import Sybil, Serie

model = Sybil("sybil_ensemble")          # default; downloads checkpoints on first use
serie = Serie(["/path/slice1.dcm", ...]) # one CT exam
prediction = model.predict([serie])      # prediction.scores -> [[p1..p6]]
```

---

## 2. Data & Training Setup

| Dataset | Module | Role |
|---|---|---|
| NLST | `sybil/datasets/nlst.py` (`NLST_Survival_Dataset`) | Primary train/dev/test dataset, built from an NLST metadata JSON |
| MGH | `sybil/datasets/mgh.py` (`MGH_Dataset`, `MGH_Screening`) | External validation cohorts against MGH's own metadata schema |
| Generic CSV | `sybil/datasets/validation.py` (`CSVDataset`, registered as `"validation"` in `sybil/utils/helpers.py:5-12`) | Bring-your-own-cohort path driven by a CSV manifest |
| Risk factors | `sybil/datasets/nlst_risk_factors.py` (`NLSTRiskFactorVectorizer`) | One-hot encoding of clinical risk factors |
| Shared helpers | `sybil/datasets/utils.py` | `VOXEL_SPACING`, `CENSORING_DIST`, `order_slices`, annotation-mask scaling |

### 2.1 Global constants (`sybil/datasets/utils.py`)

```python
IMG_PAD_TOKEN  = "<PAD>"                          # :8
VOXEL_SPACING  = (0.703125, 0.703125, 2.5)        # :9  — mm; the canonical resample target
CENSORING_DIST = {"0": 0.98519, "1": 0.97483, "2": 0.96599,
                  "3": 0.95873, "4": 0.95236, "5": 0.94619}   # :10-17 (values truncated here)
```

`CENSORING_DIST` is a baked-in Kaplan–Meier survival curve used as a fallback; at inference the real one is read from the checkpoint (§7.3).

### 2.2 NLST metadata JSON schema

`NLST_Survival_Dataset` reads a JSON list from `args.dataset_file_path` (`sybil/datasets/nlst.py:82`). Each element is one **patient**:

```
{ "pid": ..., "split": "train"|"dev"|"test",
  "pt_metadata": { <column>: [ <value> ], ... },      # column -> list; always indexed [0]
  "accessions": [                                      # one entry per exam
     { "exam": ..., "accession_number": ..., "screen_timepoint": 0|1|2,
       "date": ..., "abnormalities": ...,
       "image_series": { "<SeriesInstanceUID>": {
             "paths": [...], "img_position": [...],   # z-locations, consumed at nlst.py:257
             "pixel_spacing": [x, y], "slice_thickness": ...,
             "series_data": { "reconthickness": [...], "study_yr": [...],
                              "manufacturer": [...], "studyuid": [...],
                              "imageclass": [...], "imagetype": [...] } } } } ] }
```

The `pt_metadata` keys actually read include `scr_days{0,1,2}`, `candx_days`, `fup_days`, `cancyr`, `cen`, `age`, `smokeage`, `age_quit`, `smokeyr`, `smokeday`, `cigsmok`, `educat`, `race`, `ethnic`, `weight`, `height`, `gender`, `diagcopd`, the 14 tumour-location keys (`locrup`, `loclup`, … `locunk`, `nlst.py:364-366`), 15 prior-cancer keys (`cancblad` … `canctran`, `:421-437`), and any key prefixed `fam` for family history (`:439-441`).

This JSON is produced by `scripts/data/create_nlst_metadata_json.py` (§2.8).

> **Schema mismatch to know about**: the generator emits a `slice_location` key (`create_nlst_metadata_json.py:117-125`) while the dataset reads `img_position` (`nlst.py:257`). Both are written, but only `img_position` is consumed.

### 2.3 Exclusion rules

Applied per series in `skip_sample` (`nlst.py:218-251`). A series is dropped if **any** condition holds:

| # | Condition | Threshold / source |
|---|---|---|
| 1 | Localizer / scout scan | `imageclass[0] == 0`, or `"LOCALIZER"`/`"TOP"` in `imagetype[0]` (`:348-354`) |
| 2 | Reconstruction thickness not allowed | `reconthickness[0] not in args.slice_thickness_filter`, when set (`:224-227`) |
| 3 | Unusable label metadata | not (`scr_days{yr}[0] > -1` and (`candx_days[0] > -1` or `fup_days[0] > -1`)) (`check_label`, `:314-320`) |
| 4 | Invalid derived label | `y == -1` or `time_at_event < 0` (`:236`) |
| 5 | Too few slices | `len(paths) < args.min_num_images` (default `0`, `parsing.py:258-263`) |

Independently, `get_thinnest_cut` (`:193-216`) keeps only **one** series per exam — the one with the most slices (thinnest reconstruction), preferring an annotated series when annotations exist. Test-split series are further restricted to those present in the Google/Shetty splits pickle (`:161-172`).

At the `Serie` level (inference path) the checks are simpler — `_check_valid` (`sybil/serie.py:255-277`) raises `ValueError` on missing slice thickness, thickness `> 5` mm, or missing voxel spacing.

### 2.4 Label & censoring construction

This is a **discrete-time survival** problem, not binary classification. `NLST_Survival_Dataset.get_label` (`nlst.py:322-346`):

```python
days_to_cancer      = candx_days[0] - scr_days{screen_timepoint}[0]
years_to_cancer     = days_to_cancer // 365   if candx_days[0] > -1  else 100
years_to_last_fup   = (fup_days[0] - scr_days[0]) // 365

y     = years_to_cancer < max_followup
y_seq = zeros(max_followup);  if y: y_seq[years_to_cancer:] = 1

time_at_event = years_to_cancer                        if y
                else min(years_to_last_fup, max_followup - 1)
y_mask = [1] * (time_at_event + 1) + [0] * (max_followup - time_at_event - 1)
```

- `y_seq` is the **cumulative** target: 1 from the diagnosis year onward. This is what makes the monotone `Cumulative_Probability_Layer` (§3.6) the natural head.
- `y_mask` encodes censoring: years beyond a patient's actual follow-up contribute **zero** loss. A patient followed 2 years with no cancer supervises years 1–3 only.
- `magic 100` marks "never diagnosed" so the `< max_followup` test fails cleanly.

`Serie.get_label` (`sybil/serie.py:88-123`) implements the equivalent for the inference/eval path, returning a `Label(y, y_seq, y_mask, censor_time)` NamedTuple (`serie.py:23-27`).

### 2.5 `__getitem__` output contract

`CT_ITEM_KEYS` (`nlst.py:31-41`) lists the pass-through keys:

```python
["pid", "exam", "series", "y_seq", "y_mask", "time_at_event",
 "cancer_laterality", "has_annotation", "origin_dataset"]
```

A batch item always has `x` (the volume) and `y`, plus whichever `CT_ITEM_KEYS` are present, plus:

- `volume_annotations`, `annotation_areas`, `image_annotations`, `has_annotation` when `use_annotations` (`:565-575`)
- `risk_factors` when `use_risk_factors`

On any exception `__getitem__` warns and returns `None` (`:587-588`) — the custom collate in `sybil/utils/loading.py` is what tolerates that.

`CSVDataset.__getitem__` returns a narrower fixed set: `x, y, y_seq, y_mask, time_at_event, exam` (`validation.py:185-192`).

### 2.6 Class balancing

Cancer cases are rare, so the dataset computes per-sample weights (`nlst.py:108-118`):

```python
label_counts    = Counter(y over dataset)
weight_per_label = 1.0 / len(label_counts)          # equal mass per class
label_weights   = {label: weight_per_label / count for label, count in label_counts.items()}
self.weights    = [label_weights[d["y"]] for d in self.dataset]
```

`self.weights` is consumed by `DistributedWeightedSampler` (`sybil/utils/sampler.py`) so each epoch draws a class-balanced stream even under multi-GPU DDP.

### 2.7 Annotations

The annotation JSON is `{series_id: {image_basename: [{x, y, width, height, user}, ...]}}`, with coordinates normalized to `[0, 1]`. Loaded by `get_ct_annotations` (`nlst.py:515-550`); required when `--use_annotations` is set (`:94-100`).

`get_scaled_annotation_mask` (`datasets/utils.py:34-93`) rasterizes boxes into an `args.img_size` mask, computing **fractional coverage** at partial-overlap edges (`dx_left`, `dx_right`, `dy_top`, `dy_bottom`) rather than a hard binary fill, then normalizes by `mask.sum()`. `get_scaled_annotation_area` (`:96-105`) reduces this to a per-slice area fraction. These become the KL targets in §7.2.

### 2.8 `files/*.csv` — schema templates, not data

Both files in `files/` are **header-only templates with blank placeholder rows** — zero populated rows, no patient data. Nothing in the repo references them by filename; they document the expected schemas by convention.

`files/lung_cancer_dataset.csv` — 10 columns, consumed in spirit by `CSVDataset.parse_csv_dataset` (`validation.py:64-115`):

| Column | Meaning |
|---|---|
| `patient_id` | Patient identifier; also the grouping key for splits |
| `exam_id` | Exam identifier |
| `series_id` | Series identifier — the unit one `Serie` is built from |
| `exam_date` | Read by the template, not used by `CSVDataset` |
| `ever_has_future_cancer` | Truthy → positive label |
| `years_to_cancer` | Becomes `censor_time` when positive |
| `years_to_last_negative_followup` | Becomes `censor_time` when negative (`validation.py:136`) |
| `file_path` | Path to one slice (one row per slice) |
| `slice_position` | z-position used to order slices |
| `split` | `train` / `dev` / `test` |

> **Template quirk**: seven headers are quoted *with a leading space* (`" exam_id"`, `" series_id"`, …). `parse_csv_dataset` strips non-ASCII characters from keys (`validation.py:94`) but not leading whitespace, so the template as shipped would `KeyError` against `DictReader`. Strip the spaces before use.

`files/lung_cancer_metadata.csv` — 15 columns; the 13 after `patient_id`/`exam_id` map 1:1 onto `NLSTRiskFactorVectorizer.risk_factor_transformers` (`nlst_risk_factors.py:30-42`): `age`, `race`, `ethnicity`, `gender`, `education_level`, `weight`, `height`, `binary_family_history`, `copd`, `is_current_smoker`, `smoking_duration`, `smoking_intensity`, `years_since_quit_smoking`.

The vectorizer supports exactly 11 keys and raises `"Risk factor key '{}' not supported."` (`:49`) otherwise. Encodings are one-hot over cutoffs: `age` `[55,60,65,70,75]` → 6 features, `weight` `[155,180,210]` lbs → 4, `height` `[65,68,71]` in → 4, `race` → 6, the binary factors → 2 each. Sentinels: `MISSING_VALUE = -1`, `HASNT_HAPPENED_VALUE = -5`, `NEGATIVE_99 = -99` (`:7-25`). Default `--risk_factor_keys` is `[]` (`parsing.py:239-243`), i.e. off.

### 2.9 Data preparation scripts

- `scripts/data/create_nlst_metadata_json.py` — joins an OncoData DICOM-header JSON with three NLST CSVs (`..._ct_ab_...`, `..._prsn_...`, `..._ct_image_info_...`) plus a Google-splits xlsx, and emits the metadata JSON of §2.2. Resumable: reloads and appends to an existing output (`:33-38`, `:85-106`). Series joined on `(pid, study_yr, seriesinstanceuids)` (`:51`), patient metadata on `pid` (`:55`), all CSVs `fillna(-1)` (`:44-46`). `SPLIT_PROBS = [0.8, 0.2]` for train/dev (`:11`), with Google test PIDs forced to `test` (`:145-146`). All default input paths are absolute MIT filesystem paths.
- `scripts/data/parse_mdai_annotations.py` — converts an MD.ai export into the annotation JSON of §2.7. Boxes normalized by image `width`/`height` (`:44-52`). `INCLUDE_AFTER = 10-08-2020` drops earlier annotations (`:41`, `:70-72`). For series whose reviewer comment contains `'FF'`, only annotations by a user matching `'fintelmann'` are kept (`:82-86`). Reads a hard-coded comments CSV at import time (`:42`).

Neither script is runnable outside the original MIT environment without editing paths.

### 2.10 Adapter datasets

`MGH_Dataset` / `MGH_Screening` (`sybil/datasets/mgh.py`) mirror the NLST logic against MGH's schema: a `DEVICE_ID` map for scanner manufacturers (`:8-17`, note `SIEMENS` and `Siemens Healthcare` both map to `3`), thickness compared as a scalar rather than a set (`:139-143`), labels from `cancer_cohort_yes_no` and `diff_days_exam_lung_cancer_diagnosis` (`:163-185`), and `MGH_Screening` additionally carrying `lung_rads` from the `"LR Score"` column (`:344-357`).

**These adapters are stale and will not run as-is:** `mgh.py:98` uses `np.int` (removed in NumPy ≥ 1.24, and NumPy is pinned to 1.24.1), `mgh.py:315` calls `order_slices(..., reverse=True)` but the shared helper takes no `reverse` kwarg, and in `validation.py` line `:99` reads a nonexistent column `'fileslice_position_path'` for the 2nd+ slice of a series while `:129` references an unset `self.metadata_json`. Treat `NLST_Survival_Dataset` as the only exercised training path and `CSVDataset` as a starting point requiring repair.

---

## 3. Model Architecture

### 3.1 Input pipeline — `Serie` (`sybil/serie.py`)

`Serie` wraps a list of DICOM (or PNG) paths as one CT exam and is the boundary between raw files and model-ready tensors.

- `_load_metadata()` (`:171-223`) reads `ImagePositionPatient`, `SliceThickness`, `PixelSpacing`, `Manufacturer` and calls `order_slices()` to sort along z.
- `_check_valid()` (`:255-277`) rejects missing thickness, thickness `> 5` mm, or missing voxel spacing. (The docstring at `:266-268` also claims it checks for a missing label — it does not.)
- `get_volume()` (`:141-169`, `@functools.lru_cache`): loads slices → `(T,C,H,W)` → permute to `(C,T,H,W)` → wrap as `tio.ScalarImage` with `affine=diag(voxel_spacing)` → `tio.Resample(VOXEL_SPACING)` → `tio.CropOrPad((256,256,200), padding_mode=0)` → permute back → `unsqueeze_(0)`.
- `get_raw_images()` (`:136`) loads slices with no augmentation for visualization; it builds a fresh loader each call and is **not** cached.

Inference-time geometry is hardcoded in `_load_args` (`:239-252`), independent of `parsing.py`:

```python
img_size=[256,256], num_images=200, num_chan=3,
img_mean=[128.1722], img_std=[87.1849],
cache_path=None, use_annotations=False,
fix_seed_for_multi_image_augmentations=True, slice_thickness_filter=5
```

### 3.2 Loaders & augmentation

- `sybil/loaders/abstract_loader.py` — slice loading with an optional **disk cache**: `.npy` arrays keyed by `md5(image_path)` under `cache_dir/<attr_key>/<parent_dirname>/<md5>.npy` (`:21-25`, `:89-90`, `:104`). `split_augmentations_by_cache` (`:28-58`) builds progressively longer cache keys from consecutive `cachable()` augmentations, reversed so the latest cache point is tried first. Corrupt entries are deleted with a warning (`:202-208`).
  **The cache is inert at inference**: `Serie._load_args` sets `cache_path=None` (`serie.py:247`), so `use_cache=False` and all augmentations run every time (`abstract_loader.py:139-145`).
- `sybil/loaders/image_loaders.py`:
  - `OpenCVLoader` — grayscale PNG via `cv2.imread(path, 0)` (`:13-17`).
  - `DicomLoader` — `dcmread` → `apply_modality_lut` → `apply_windowing` → `arr // 256`. Windowing is **hardcoded lung CT**: `window_center = -600`, `window_width = 1500` (`:27-28`); `apply_windowing` (`:45-76`) assumes `bit_size=16` (`y_max = 65535`) and mutates in place.
- `sybil/augmentations.py` — train: `Scale_2d → Rotate_Range(±20°) → ToTensor → Force_Num_Chan_Tensor_2d → Normalize_Tensor_2d`; dev/test drops the rotation.

Loader selection is `sybil/utils/loading.py:161-192` (`"dicom"` / `"png"`, else `NotImplementedError`). There is **no** remote/Ark loader in the package — Ark is an external HTTP server exercised only from `examples/remote_ark.py` and the regression test.

### 3.3 Encoder — `SybilNet` (`sybil/models/sybil.py:9-55`)

```python
encoder = torchvision.models.video.r3d_18(pretrained=True)          # :15
self.image_encoder = nn.Sequential(*list(encoder.children())[:-2])  # :16 — drops avgpool + fc
```

`VideoResNet.children()` is `[stem, layer1..layer4, avgpool, fc]`, so `[:-2]` keeps the stem and four residual stages. The output is therefore a **spatial-temporal feature map**, not a vector — which is what makes attention pooling possible. `hidden_dim = 512` is hardcoded (`:13`) to match ResNet-18's final channel count.

Pretrained weights load **unconditionally** — there is no `--pretrained` flag anywhere in `parsing.py`.

`forward()` (`:27-35`): `image_encoder` → `aggregate_and_classify` → output dict with `activ` (raw feature map), all pooling outputs, `hidden`, `logit`, and `prob = logit.sigmoid()`.

`aggregate_and_classify` (`:37-44`) is the only place regularization happens: `hidden` is ReLU'd then passed through the single `nn.Dropout(p=args.dropout)` (`:20-21`, `:40-41`) before the survival head. There is no dropout inside the encoder or the pooling layer.

`RiskFactorPredictor(SybilNet)` (`:58-80`) is **dead code**, not a working auxiliary variant: `forward` references `self.args` (never assigned on `SybilNet`) and `__init__` reads `args.hidden_dim` (not an argparse argument), so instantiating and running it raises. The paper's finding that clinical risk factors are recoverable from the image (`knowledge-base.md:30`) is not reproducible from this class as shipped.

### 3.4 Multi-level attention pooling — `MultiAttentionPool` (`sybil/models/pooling_layer.py`)

The topology is **two two-stage cascades plus one global pool** — not five parallel branches. Each cascade first reduces spatially *within* each slice, then attends *across* slices (`:28-32`):

```
x (B, 512, T, W, H)
├── branch 1: Simple_AttentionPool_MultiImg ──► Simple_AttentionPool   ──► hidden (B,512)
│              (spatial attention per slice)     (attention over slices)
├── branch 2: PerFrameMaxPool ────────────────► Conv1d_AttnPool        ──► hidden (B,512)
│              (spatial max per slice)           (Conv1d k=11 → slice attention)
└── branch 3: GlobalMaxPool ─────────────────────────────────────────► hidden (B,512)
```

| Sub-module | Role | Output keys / shapes |
|---|---|---|
| `Simple_AttentionPool_MultiImg` (`:151-185`) | Linear(512→1) + softmax over `W*H`, per slice independently | `image_attention (B,T,W*H)` (log-softmax, `:178`), `multi_image_hidden (B,C,T)` (`:183`), `hidden (B,T*C)` (`:184`) |
| `Simple_AttentionPool` (`:116-148`) | Linear(512→1) + softmax over the slice axis | `volume_attention (B,N)` (log-softmax, `:143`), `hidden (B,C)` (`:147`) |
| `PerFrameMaxPool` (`:70-90`) | Per-slice max over spatial dims — strongest localized response regardless of position | `multi_image_hidden (B,C,T)` **only** |
| `Conv1d_AttnPool` (`:93-113`) | `Conv1d(512, 512, kernel=11, stride=1, padding=5, bias=False)` over slices, then `Simple_AttentionPool` | `hidden (B,C)`, `volume_attention (B,N)` |
| `GlobalMaxPool` (`:50-67`) | Max over all spatial + temporal positions | `hidden (B,C)` |

Attention outputs are **log-softmax**, i.e. log-probabilities. That is deliberate: the losses feed them directly into `F.kl_div` (which expects log-inputs), and consumers must `exp()` them — as `collate_attentions` does (`visualization.py:16-17`).

**Fusion** (`:41-45`):

```python
multi_image_hidden = cat([image_pool1.multi_image_hidden,
                          image_pool2.multi_image_hidden], dim=-2)      # (B, 1024, T)
output['multi_image_hidden'] = multi_img_hidden_fc(...)                  # Linear 1024 -> 512

hidden = cat([volume_pool1.hidden, volume_pool2.hidden, maxpool_hidden], dim=-1)  # (B, 1536)
output['hidden'] = hidden_fc(hidden)                                    # Linear 1536 -> 512
```

Three things about this are worth internalizing:

1. **Only `hidden_fc` feeds the prediction.** Three tensors are concatenated → 1536 → 512, and that 512-d vector is the sole input to the survival head.
2. **`multi_img_hidden_fc` is a dead end.** `output['multi_image_hidden']` is never consumed by `hidden`, by the classifier, or by any loss in `sybil/utils/losses.py`. Those ~525k parameters receive no gradient and are effectively frozen at init in the released checkpoints.
3. **`hidden_1` is silently overwritten.** The suffixing loop (`:34-36`) iterates `[(image_pool_out1, 1), (volume_pool_out1, 1), (image_pool_out2, 2), (volume_pool_out2, 2)]` and both pools in each pair emit a `hidden` key. So `output['hidden_1']` is written as the image pool's `(B, T*C)` (`:184`) and then immediately replaced by the volume pool's `(B, C)` (`:147`). Anyone reading `hidden_1` gets the volume-pool vector.

Also note the asymmetry: **`image_attention_2` does not exist**, because `PerFrameMaxPool` emits no attention (`:88-90`). This is why `Sybil.predict` surfaces only `image_attention_1` and `volume_attention_1` (`model.py:293-302`), and it has a direct consequence for the annotation loss (§7.2).

### 3.5 Why this pooling design

Each branch answers a different question, and the ablation-free fusion lets the model use whichever is informative:

- **Attention cascade (branch 1)** — "which pixels in which slices matter?" Soft, differentiable, and directly interpretable; it is the only branch whose intermediate maps are exposed for visualization.
- **Max cascade (branch 2)** — "is a strong suspicious response present anywhere?" Max pooling is translation-invariant within a slice and resists dilution by surrounding normal tissue, so a small focal nodule is not averaged away. The subsequent `Conv1d(kernel=11)` then models a *neighbourhood* of slices, capturing the fact that a real 3D lesion appears across several consecutive slices while noise does not.
- **Global max (branch 3)** — a cheap unconditional escape hatch: the single strongest activation in the whole volume, bypassing both attention layers.

### 3.6 Survival head — `Cumulative_Probability_Layer`

A **discrete-time hazard model** guaranteeing monotone non-decreasing cumulative risk (`sybil/models/cumulative_probability_layer.py:5-33`):

```python
self.hazard_fc      = nn.Linear(512, max_followup)   # :9
self.base_hazard_fc = nn.Linear(512, 1)              # :10
mask = torch.tril(torch.ones([max_followup, max_followup]), diagonal=0)
self.register_parameter("upper_triagular_mask",
                        nn.Parameter(torch.t(mask), requires_grad=False))  # :12-15

hazards         = relu(hazard_fc(x))                  # (B, T), forced >= 0
expanded        = hazards.unsqueeze(-1).expand(B, T, T)
masked          = expanded * self.upper_triagular_mask
cum_prob        = masked.sum(dim=1) + base_hazard_fc(x)   # (B, T)
```

So `cum_prob[b, t] = Σ_{i ≤ t} hazard_i + base_hazard`. Because every hazard is ReLU'd non-negative and cumulatively summed, `cum_prob` cannot decrease across years — the domain constraint that cumulative cancer probability is monotone is enforced *architecturally*, not learned.

Two details: `base_hazard` is a **single scalar** broadcast to all six years (an intercept, not a per-year bias), and the parameter name `upper_triagular_mask` carries a typo that is baked into every released state dict — do not "fix" it without a migration.

### 3.7 End-to-end shape trace

Derived from r3d_18's strides for the released geometry (stem `stride=(1,2,2)`; layers 2–4 each `stride=(2,2,2)`; layer 1 stride 1):

| Stage | Shape |
|---|---|
| `Serie.get_volume()` output | `(1, 3, 200, 256, 256)` |
| after `stem` | `(1, 64, 200, 128, 128)` |
| after `layer1` | `(1, 64, 200, 128, 128)` |
| after `layer2` | `(1, 128, 100, 64, 64)` |
| after `layer3` | `(1, 256, 50, 32, 32)` | 
| after `layer4` = `activ` | **`(1, 512, 25, 16, 16)`** |
| `image_attention_1` | `(1, 25, 256)` — 25 slices × 16·16 positions |
| `volume_attention_1` / `_2` | `(1, 25)` |
| `multi_image_hidden_1` / `_2` | `(1, 512, 25)` |
| `hidden_1` (after overwrite, §3.4) | `(1, 512)` |
| `maxpool_hidden` | `(1, 512)` |
| fused `hidden` | `(1, 512)` |
| `logit` → `prob` | `(1, 6)` |

Two consequences of `T = 25`:

- `Conv1d_AttnPool`'s kernel of 11 spans nearly half the 25-slice sequence — a very wide receptive field relative to the sequence length.
- The volume-annotation loss interpolates because `N = 25 ≠ args.num_images = 200`, so that branch always fires during training (`losses.py:110-111`).

### 3.8 Flow summary

```
Serie (DICOM/PNG list)
  → get_volume(): tio.Resample(VOXEL_SPACING) + CropOrPad → (1, 3, 200, 256, 256)
  → SybilNet.image_encoder (r3d_18 minus avgpool/fc) → activ (1, 512, 25, 16, 16)
  → MultiAttentionPool → hidden (1, 512) + attention maps
  → ReLU → Dropout → Cumulative_Probability_Layer → logit (1, 6)
  → sigmoid → prob (1, 6)                        [raw, uncalibrated]
  → mean over 5 ensemble members                 [still uncalibrated]
  → SimpleClassifierGroup per year               → calibrated risk scores
```

---

## 4. Hyperparameters (`sybil/parsing.py`)

Defaults for the arguments that shape the released model. Training args reach inference via `checkpoint["args"]`.

| Argument | Default | Line |
|---|---|---|
| `--dataset` | `"nlst"` | 133-144 |
| `--img_size` | `[256, 256]` | 145-151 |
| `--num_chan` | `3` | 152-154 |
| `--img_mean` / `--img_std` | `[128.1722]` / `[87.1849]` | 155-168 |
| `--num_classes` | `6` | 194-196 |
| `--split_probs` | `[0.6, 0.2, 0.2]` | 218-223 |
| `--max_followup` | `6` | 227-229 |
| `--use_risk_factors` / `--risk_factor_keys` | `False` / `[]` | 232-243 |
| `--resample_pixel_spacing_prob` | `1` | 246-251 |
| `--num_images` | `200` | 252-257 |
| `--min_num_images` | `0` | 258-263 |
| `--use_annotations` | `False` | 277-282 |
| `--annotation_loss_lambda` | `1` | 287-292 |
| `--image_attention_loss_lambda` | `1` | 293-298 |
| `--volume_attention_loss_lambda` | `1` | 299-304 |
| `--primary_loss_lambda` | `1.0` | 307-312 |
| `--adv_loss_lambda` | `1.0` | 313-318 |
| `--batch_size` | `32` | 321-326 |
| `--init_lr` | `0.001` | 327-332 |
| `--dropout` | `0.25` | 333-338 |
| `--optimizer` | `"adam"` | 339-341 |
| `--momentum` | `0` (SGD only) | 342-344 |
| `--lr_decay` | `0.1` | 345-350 |
| `--weight_decay` | `0` | 351-356 |
| `--patience` | `5` | 365-370 |
| `--tuning_metric` | `"c_index"` | 377-382 |
| `--num_workers` | `8` | 404-409 |

Notes:

- **Stale help strings**: `--batch_size`'s help says 128 while the default is 32; `--lr_decay`'s says 0.5 while the default is 0.1. Trust the defaults, not the text.
- **No `--pretrained`** — r3d_18 ImageNet/Kinetics weights load unconditionally (`models/sybil.py:15`).
- `args.lr = args.init_lr` is set post-parse (`:460`); `accelerator="ddp"` only when more than one GPU is visible, with `replace_sampler_ddp=False` either way (`:462-469`).
- `Trainer.add_argparse_args` merges all PyTorch Lightning flags (`:455`), so `max_epochs`, `gpus`, and `precision` come from **Lightning's** defaults — they are not documented in this file, and the repo does not record the exact values used for the released checkpoints.
- `--primary_loss_lambda` is misleading: it is consumed only by `get_risk_factor_loss`, never by the survival loss. See §7.1.

---

## 5. Calibration (`sybil/models/calibrator.py`)

Raw sigmoid outputs are recalibrated per follow-up year using **isotonic regression, reimplemented from scratch** so that inference has no scikit-learn dependency (sklearn is a `train`-extra only, `setup.cfg:69-73`).

- `SimpleIsotonicRegressor.transform()` (`:78-82`): a linear pre-transform (`coef`, `intercept`, from the original sklearn `LogisticRegression` base estimator), clip to `[x_min, x_max]`, then piecewise-linear `np.interp` over knots `(x0, y0)` copied from a fitted sklearn isotonic calibrator.
- `SimpleClassifierGroup` (`:15-66`) holds one regressor per CV fold — mirroring `sklearn.calibration.CalibratedClassifierCV` — and averages their outputs, reproducing `predict_proba` without sklearn at runtime.
- Serialized as JSON keyed `"Year1"` … `"Year6"`, loaded via `SimpleClassifierGroup.from_json_grouped()`.
- `export_calibrator` / `export_by_name` / `export_all_default_calibrators` (`:116-146`) are one-time converters from legacy sklearn pickles; `run_test_calibrations` (`:149-164`) supports regression-testing that conversion.

Calibration is applied **once, to the ensemble mean** — not per member (§6).

---

## 6. Ensemble & Inference API (`sybil/model.py`)

- **Aliases & checkpoints**: `NAME_TO_FILE` (`:19-67`) maps `sybil_base`, `sybil_1`…`sybil_5`, `sybil_ensemble` to MD5-named `.ckpt` basenames plus a calibrator id. `sybil_base` and `sybil_1` are the **same checkpoint**. `sybil_ensemble` (the default) is the five single models with its own calibrator.
- **Weight source**: a GitHub release zip — `CHECKPOINT_URL = os.getenv("SYBIL_CHECKPOINT_URL", ".../releases/download/v1.5.0/sybil_checkpoints.zip")` (`:69`), extracted into `~/.sybil/` (`:137-138`). The `google_checkpoint_id` / `google_calibrator_id` fields in `NAME_TO_FILE` are **vestigial** (see the comment at `:18`); Google Drive is no longer the fetch path.
- **Checkpoint loading** (`load_model`, `:194-223`): `torch.load(path, map_location="cpu", weights_only=False)`; reads `args.max_followup` (`:210`) and `args.censoring_distribution` (`:211`) onto the wrapper while passing the whole `args` to `SybilNet(args)` (which also consumes `args.dropout`); strips the Lightning `"model."` prefix; calls `model.eval()` — so dropout is inert at inference.
- **Prediction** (`_predict`, `:248-305`): per-serie, batch size 1, under `no_grad`. It uses `out["logit"].sigmoid()`, recomputing what `out["prob"]` already contains. Returned attentions are only `image_attention_1`, `volume_attention_1`, and `hidden` (`:293-302`).
- **Ensembling** (`:340-349`): a plain arithmetic mean of the five members' **uncalibrated** sigmoid scores, then a single isotonic calibration of that mean (`_calibrate`, `:225-246`). Attentions are **stacked**, not averaged (`:353-360`), so each member's map stays separately inspectable.
- **Evaluation** (`:364-412`): requires labels on every `Serie`, builds a `Namespace(max_followup, censoring_distribution)` from the checkpoint, and calls `get_survival_metrics` for per-year AUC plus c-index.
- **Device management** (`to()` / `_pick_device`, `:414-448`): supports a fixed device or "most-free-GPU" auto-selection, estimating the ensemble footprint at ~9× parameter bytes and only relocating if the current GPU looks shared. Useful for multi-process inference without cross-process coordination.
- `_torch_set_num_threads()` (`:118-131`) defaults CPU threads to `min(8, cpu_count())`.

---

## 7. Training Pipeline

Training lives in `scripts/`, outside the installed package. `scripts/train.py` defines `SybilLightning`, a PyTorch Lightning module wrapping `SybilNet`.

### 7.1 Loss composition

```python
def get_loss_functions(self, args):              # train.py:243-248
    loss_fns = [losses.get_survival_loss]
    if args.use_annotations:
        loss_fns.append(losses.get_annotation_loss)
    return loss_fns
```

Losses are **summed unweighted** (`train.py:65-73`). The only weights in the system are the lambdas *inside* `get_annotation_loss` — so the survival loss enters the total with an effective weight of exactly 1.0. `--primary_loss_lambda` does not apply to it.

**`get_survival_loss`** (`sybil/utils/losses.py:17-26`):

```python
loss = F.binary_cross_entropy_with_logits(
           logit, y_seq.float(), weight=y_mask.float(), reduction='sum'
       ) / torch.sum(y_mask.float())
```

This is **per-year binary cross-entropy on the cumulative logits, masked by follow-up availability, and normalized by the number of supervised year-slots** — not a Cox partial likelihood. Combined with the monotone head (§3.6), the model learns non-negative yearly hazards whose cumulative sums match the cumulative label `y_seq` over exactly the years each patient was actually followed.

`get_cross_entropy_loss` (`:7-14`), `get_risk_factor_loss` (`:163-185`), and `discriminator_loss` (`:187-198`) exist but are never registered for Sybil — the last calls `model.discriminator`, which `SybilNet` does not have.

### 7.2 Annotation loss (opt-in)

`get_annotation_loss` (`losses.py:29-160`) loops `attn_num in [1, 2]` and assembles up to four terms per iteration:

| Term | Mechanism | Weight |
|---|---|---|
| Image KL | Gold masks `F.interpolate(..., mode="area")` → zeroed where `has_annotation` is false → renormalized per slice → `F.kl_div(pred_attn, gold)` masked to `gold > 0`, divided by the number of annotated slices (`:68-76`) | `image_attention_loss_lambda` |
| Image side | `exp(image_attention)` split into left/right halves at `W//2`, cross-entropy against `cancer_laterality[:,1]`, masked to unilateral cases (`:78-104`) | `image_attention_loss_lambda` |
| Volume KL | Gold = `annotation_areas` normalized per volume, linearly interpolated when `N != args.num_images`, masked KL (`:106-133`) | `volume_attention_loss_lambda` |
| Volume side | Laterality term, guarded by `isinstance(side_attn, torch.Tensor)` (`:135-158`) | `volume_attention_loss_lambda` |

The whole sum is scaled by `annotation_loss_lambda` at return (`:160`).

**With defaults, exactly five terms are active, not eight** — because `image_attention_2` does not exist (§3.4), iteration `n=2` skips both its image terms, and its volume *side* term is skipped too (that guard depends on the image branch of the same iteration having run). The surviving set is: image KL₁, image side₁, volume KL₁, volume side₁, volume KL₂ — each at λ=1.

Annotations therefore supervise attention **purely through the loss**. There is no annotation head, no gating, and nothing about the architecture changes when they are enabled.

Two quirks in this function: the image side loss applies `log_softmax` to values that are already summed probabilities (`:78-104`), and with `N=25` vs `num_images=200` the volume interpolation branch always fires.

### 7.3 Optimization & checkpointing

- **Optimizer** (`train.py:186-207`): Adam / Adagrad / SGD over `requires_grad` params, `lr=args.lr`, `weight_decay=args.weight_decay`; momentum for SGD only.
- **Scheduler** (`:209-219`): `ReduceLROnPlateau(patience=args.patience, factor=args.lr_decay)` on `val_{tuning_metric}`, epoch interval, `mode="max"` for `c_index`.
- **Checkpointing** (`:336-347`): `save_top_k=1`, `save_last=True`, monitoring `val_c_index` with `mode="max"`.
- **Censoring distribution**: `args.censoring_distribution = metrics.get_censoring_dist(train_dataset.dataset)` is computed on the **train** split before the module is built (`:365`, and `:398` for test). Because it lives on `args`, it is serialized inside `checkpoint["args"]` and read back at inference by `sybil/model.py:211` — this is how a training-set survival curve reaches `evaluate()`.
- `step()` passes only `batch["x"]` to the model (`:53`), so `SybilNet.forward`'s `batch=None` parameter is never used.
- Metrics run every epoch (`:40-44`). Note `get_classification_metrics` is called on survival probabilities with `num_classes == 6`, so it reports only accuracy — its precision/recall/AUC block requires `num_classes == 2` (`metrics.py:29`).

Supporting modules: `sybil/utils/sampler.py` (`DistributedWeightedSampler`, fed by §2.6's weights), `sybil/utils/loading.py` (`get_sample_loader` + custom `default_collate` tolerating `None` items), `sybil/utils/helpers.py` (`get_dataset` name dispatch).

Baseline comparator: `scripts/plcom2012/` implements and evaluates **PLCOm2012**, the clinical risk-factor model Sybil is compared against in the paper. `NLST_for_PLCO` / `NLST_for_PLCO_Screening` (`nlst.py:656-758`) provide the risk-factor-only sample views it needs.

---

## 8. Metrics (`sybil/utils/metrics.py`)

`get_survival_metrics` (`:47-70`) emits, for each `followup in range(max_followup)`, the keys `{k}_year_auc`, `{k}_year_apscore`, `{k}_year_prauc`, then a single `c_index`. If there are no positives it returns `-1.0` rather than raising.

**Case inclusion at each horizon** (`compute_auc_at_followup`, `:121-151`) is the subtle part — censoring means not every patient can be scored at every year:

```
positive  iff  gold and censor_time <= followup and censor_time > fup_lower_bound   # default -1
negative  iff  censor_time >= followup
```

Only included cases are scored, using `prob_arr[followup]`. A patient censored at year 2 is therefore a valid negative for years 1–2 but is **excluded** from the year-5 AUC — you cannot know they were cancer-free that long. `get_risk_metrics` (`:102-118`) is the same computation with `fup_lower_bound=0` and `_risk_` key names. Failures log a warning and return `-1.0`.

`get_censoring_dist` (`:154-166`) fits a Kaplan–Meier curve (via `lifelines`) on the train split's `time_at_event`/`y` and stores it as `{str(time): S(time)}`.

`concordance_index` (`:169-244`) is **time-dependent with IPCW weighting** — i.e. Uno's C, not Harrell's:

- Scores are inverted (`1 - probs`, `:222`) so that higher risk means shorter predicted survival.
- At each observed event time `t` the comparison uses `pred[:, int(t)]`, with a separate `_BTree` per observed time (`:316`, `:381`).
- The inverse-probability-of-censoring weight is `1 / censoring_dist[str(int(truth_time))] ** 2` (`:374`), applied to the whole tied batch (`:354-356`) rather than per pair.

---

## 9. Interpretability (`sybil/utils/visualization.py`)

`collate_attentions(attention_dict, N, eps=1e-6)` (`:8-30`) turns raw model attention into a per-slice heatmap volume:

```python
a1 = torch.exp(a1).mean(0)          # :16 — exp first: attentions are log-softmax
v1 = torch.exp(v1).mean(0)          # :17 — mean over ensemble members
attention = a1 * v1.unsqueeze(-1)   # :19 — spatial x per-slice importance
attention = attention.view(1, 25, 16, 16)                                    # :20
attention_up = F.interpolate(attention.unsqueeze(0), (N, 512, 512),
                             mode="trilinear")                               # :22-24
attention_up[attention_up <= eps] = 0.0                                      # :27-28
```

Note the ensemble is averaged in **probability space, not logit space**, and that the product of image and volume attention is what gets visualized — a pixel is highlighted only if both its slice and its position within that slice were attended to.

**The `(1, 25, 16, 16)` and `(N, 512, 512)` targets are hardcoded magic constants**, tied to `num_images=200` (→ T=25 after the encoder, §3.7) and to 512×512 source DICOMs. Change the input geometry and this reshape either throws or silently misaligns.

`build_overlayed_images(images, attention, gain=3)` (`:32-47`) composes a hardcoded `512×512×3` frame: channels 2 and 1 carry the raw grayscale CT, channel 0 carries `clip(attention[i] * gain * 256 + images[i], 0, 255)`. `visualize_attentions(series, attentions, save_directory=None, gain=3)` (`:50-83`) accepts either a list of `Serie` or a single one (`:67-68`), pulls raw pixels via `serie.get_raw_images()` (`:72`), and `save_images` (`:86-101`) writes **`.gif` animations** via `imageio.mimsave` — one per series, not per-slice PNGs.

Only two attention granularities are surfaced end-to-end: **image-level** (where within a slice) and **volume-level** (which slices matter). This is the mechanism behind the clinical-validation questions in `knowledge-base.md:87-90`.

---

## 10. CLI, Examples, Tests

### CLI — `sybil/predict.py`

Installed as `sybil-predict` (`setup.cfg:75-77`). Flags (`_get_parser`, `:19-78`):

| Flag | Default | Notes |
|---|---|---|
| `image_dir` (positional) | — | **Every** file in the directory is included |
| `--output-dir` | `sybil_result` | |
| `--return-attentions` | off | |
| `--write-attention-images` | off | Implies `--return-attentions` (`:92`) |
| `--file-type` | `auto` | `{dicom, png, auto}` |
| `--model-name` | `sybil_ensemble` | |
| `-l/--log-level` | `INFO` | |
| `--threads` | `0` | 0 = all cores; negative = PyTorch default |
| `-v/--version` | — | |

Outputs into `--output-dir`: `prediction_scores.json` (`{"predictions": scores}`, indent 2, `:130-133`), also echoed to stdout (`:168`); `attention_scores.pkl` pickling the whole prediction object when attentions are requested (`:137-139`); attention GIFs via `visualize_attentions(..., gain=3)` when `--write-attention-images` (`:141-147`). PNG input injects `VOXEL_SPACING` explicitly (`:110`).

> Latent bug in `auto` detection (`:100-105`): the code pops one extension from the set and *then* checks `len(extensions) > 1`, so a directory containing exactly two distinct extensions passes the check silently.

`scripts/run_predict_demo.sh:7-21` downloads `sybil_example.zip` and runs `sybil-predict --loglevel DEBUG --output-dir demo_prediction --return-attentions`. It does **not** pass `--write-attention-images`, so the demo produces no images.

### Examples

- `examples/local.py:15-36` — load `Sybil("sybil_ensemble")`, build a `Serie` from demo DICOMs (`examples/utils.py:21-42`, cached under `~/.sybil/sybil_example/`), predict with attentions, visualize into `sybil_attention_output`.
- `examples/remote_ark.py` — client for an external **Ark** inference server: `GET {host}/info`, asserts `modelName == "sybil"` (`:43`), then `POST {host}/dicom/files` with multipart DICOMs and `data={"data": json.dumps({"return_attentions": True})}` (`:34-48`). The response's `data.predictions[0]` is scores, `[1]` is attentions (`:54-61`); overlays are built locally (`:69`). Default host `localhost:5000`. (`:25` uses `sybil.Serie` while importing only `sybil.utils.visualization` — it works incidentally.)

### Tests (`tests/`)

The suite runs almost nothing by default:

- `tests/test_create_sybilnet.py:7-18` — the only pure unit test: builds `SybilNet(Namespace(dropout=0.1, max_followup=5))` and asserts `hidden_dim == 512` and `prob_of_failure_layer is not None`. No weights, no data.
- `tests/regression_test.py` — **all three heavy tests are gated on `SYBIL_TEST_RUN_REGRESSION == "true"`** and otherwise `pytest.skip` (`:116-118`, `:201-203`, `:324-326`):
  - `test_demo_data` (`:115-171`) — the real regression check: downloads the demo exam, runs the default ensemble, and compares the six scores against a hardcoded reference `[0.0216288…, 0.0385726…, 0.0719195…, 0.0792698…, 0.0958458…, 0.1356809…]` (`:122-129`) with `math.isclose(rel_tol=1e-6)` (`:160`), then writes attention GIFs.
  - `test_nlst_predict` (`:200-321`) — downloads 26 hardcoded NLST SeriesInstanceUIDs from TCIA, skips series with `< 20` files (`:273`), and writes `tests/nlst_predictions/nlst_predictions_{model}_v{version}.json` incrementally (resumable, `:246-251`). It **asserts nothing** — it only generates data. Optional Ark mode via `SYBIL_TEST_USE_ARK` / `SYBIL_ARK_HOST`.
  - `test_compare_predict_scores` (`:323-385`) — diffs that generated file against baseline `tests/nlst_predictions/nlst_predictions_ark_v1.4.0.json` per year with `abs_tol=1e-6`. **Neither the baseline nor the `tests/nlst_predictions/` directory exists in the checkout**, so this cannot run without first obtaining one.
  - `test_calibrator` (`:387-419`) — *not* gated, so it is the only substantive assertion that runs by default, but it downloads a calibrations JSON and may instantiate the full ensemble, comparing `predict_proba` with `np.allclose(atol=1e-10)`.
- `tests/test_one.py` is `assert True`; `tests/conftest.py` is a single commented-out import — no fixtures.

**Practical implication**: a plain `pytest` run gives essentially no coverage of the model. To validate a change, set `SYBIL_TEST_RUN_REGRESSION=true` and rely on `test_demo_data`'s hardcoded reference scores.

---

## 11. Results & Clinical Findings

All numbers in this section come from `knowledge-base.md` (i.e. the paper), not from the repository — the codebase itself ships no performance figures, and `README.md`/`CHANGELOG.md` report none.

- AUC ≈ 0.86–0.94 at 1 year, ≈ 0.74–0.81 at 6 years across NLST test / MGH / CGMH (`knowledge-base.md:37`). No per-cohort breakdown or confidence intervals are recorded in these notes.
- CGMH (different ethnicity and geography from the training data) performed comparably to the NLST test set (`:23-24`).
- Removing known-cancerous **visible** nodules from the evaluation set lowered performance but left predictive power intact (`:26-27`) — evidence Sybil is not purely a nodule detector.
- Sybil's ability to correctly lateralize (left/right) future cancer location correlates with the likelihood of a high risk score (`:28`). This is the finding the laterality loss terms of §7.2 target.
- Clinical risk factors such as smoking duration are predictable directly from the LDCT (`:30`). Note the `RiskFactorPredictor` class that would demonstrate this is non-functional as shipped (§3.3).
- Case-level utility: Sybil flagged high risk (60th risk percentile) on scans a radiologist scored Lung-RADS 1–2, i.e. low risk (`:31`).
- Interpretation caveat from the notes (`:15-16`): a low 1–6 year score is a statement about that window only, not indefinite reassurance.

Citation (`README.md:123-132`): Mikhael et al., *JCO* 2023. Training data via TCIA, DOI `10.7937/TCIA.HMQ8-J677` (`README.md:117-119`).

---

## 12. Limitations & Engineering Considerations

### Clinical / generalizability

- **Demographics and era**: NLST scans are from 2002–2004 and 92% White US patients (`knowledge-base.md:33`) — a documented open concern, only partly addressed by the CGMH validation.
- **Static calibration**: the isotonic calibrators are fit once offline and shipped as JSON per model name. They are never updated per-deployment, and they assume the calibration-set score distribution matches production data. A cohort or scanner shift degrades calibration even if discrimination (AUC) holds.
- **Calibration follows averaging**: the ensemble mean is taken on raw scores and calibrated once (`model.py:340-349`) — one calibrator per model name instead of per member.

### Input constraints

- **Fixed geometry**: every exam is forced to `(200 slices, 256×256)` by `tio.Resample` + `CropOrPad` (`serie.py:73-76`). Scans with very different slice counts or thickness are resampled to fit; atypical acquisitions lose information.
- **Slice thickness filter**: series thicker than 5 mm are rejected outright (`serie.py:250`, `_check_valid`), implicitly restricting eligible inputs.
- **Hardcoded lung windowing**: `DicomLoader` fixes `window_center=-600`, `window_width=1500` (`image_loaders.py:27-28`) and assumes 16-bit source data. Non-conventional sources need a different loader.
- **PNG ordering is the caller's responsibility** (`README.md:52-56`) — silently wrong predictions result from mis-ordered PNG input, since there is no z-position to sort by.

### Code-level gotchas

- **`multi_img_hidden_fc` is dead weight** — computed every forward pass, consumed by nothing, receives no gradient (`pooling_layer.py:41-42`, §3.4).
- **`hidden_1` is overwritten** by the suffixing loop (`pooling_layer.py:34-36`); the image pool's `(B, T*C)` never escapes.
- **`image_attention_2` does not exist**, which silently drops three of the eight nominal annotation-loss terms (§7.2).
- **`lru_cache` on an instance method**: `Serie.get_volume` (`serie.py:141`) is cached at class level and keyed on `self`, so every `Serie` ever constructed — and its ~200×256×256 volume — stays alive for the process lifetime. This is a real memory leak in long-running batch jobs.
- **The loader disk cache never activates at inference** because `Serie._load_args` sets `cache_path=None` (`serie.py:247`). It also has latent bugs on the enabled path: `abstract_loader.py:178-181` reads `input_dict["annotations"]` before `load_input` populates it, and `:209`/`:218` reference a loop variable after the loop.
- **Metadata read from an arbitrary slice**: thickness, pixel spacing, and manufacturer are taken from `dcm`, the *last* file in the unsorted input iteration (`serie.py:189-204`), not from a consistently chosen slice.
- **`RiskFactorPredictor` and the MGH/CSV adapters do not run as-is** (§3.3, §2.10).
- **State dict typo is load-bearing**: `upper_triagular_mask` (`cumulative_probability_layer.py:15`) appears in every released checkpoint.
- **`torch.load(..., weights_only=False)`** (`model.py:200`) executes arbitrary pickle payloads — fine for the official release zip, a real risk for third-party checkpoints via `SYBIL_CHECKPOINT_URL`.

### Environment & release hygiene

- `python_requires = >=3.8,<3.11` (`setup.cfg:34`) with hard pins: `torch==1.13.1`, `torchvision==0.14.1`, `numpy==1.24.1`, `pydicom==2.3.0`, `torchio==0.18.74`, `opencv-python==4.5.4.60` (`setup.cfg:39-53`). Extras: `testing` (pytest/flake8/mypy/black) and `train` (albumentations, lifelines, pytorch_lightning==1.6.0, scikit-learn). The pinned stack is old enough that a modern environment will not satisfy it without a dedicated venv.
- **Version drift**: `README.md:1` badges 1.2.0, `setup.cfg:10` declares 1.4.0, `setuptools_scm` derives the actual `__version__` from git (`pyproject.toml:5-7`, `sybil/__init__.py:11-16`), the checkpoint URL points at v1.5.0 (`model.py:69`), and the test baseline filename pins v1.4.0 (`regression_test.py:329`). Do not trust any single one as the release version.
- `CHANGELOG.md` contains only "Version 0.0.0 (development) / Initial release" — no usable history.
- License: MIT (`LICENSE.txt`).

---

## 13. Appendix — Module Index

| Path | Purpose | Status |
|---|---|---|
| `sybil/model.py` | `Sybil`: ensemble loading, checkpoint download, predict/evaluate | Core |
| `sybil/serie.py` | `Serie`: DICOM/PNG → volume tensor, label construction | Core |
| `sybil/predict.py` | `sybil-predict` CLI | Core |
| `sybil/parsing.py` | Training/CLI argument parser | Core |
| `sybil/augmentations.py` | 2D per-slice augmentation pipeline | Core |
| `sybil/models/sybil.py` | `SybilNet` | Core; `RiskFactorPredictor` is broken |
| `sybil/models/pooling_layer.py` | `MultiAttentionPool` + sub-pools | Core; `multi_img_hidden_fc` unused |
| `sybil/models/cumulative_probability_layer.py` | Monotone hazard survival head | Core |
| `sybil/models/calibrator.py` | sklearn-free isotonic calibration | Core |
| `sybil/loaders/abstract_loader.py` | Loader + disk cache framework | Cache path inert at inference |
| `sybil/loaders/image_loaders.py` | `OpenCVLoader`, `DicomLoader` | Core |
| `sybil/datasets/nlst.py` | `NLST_Survival_Dataset` + PLCO views | Primary training dataset |
| `sybil/datasets/mgh.py` | MGH cohort adapters | Stale (`np.int`, `order_slices(reverse=)`) |
| `sybil/datasets/validation.py` | `CSVDataset` — bring-your-own cohort | Has column/attribute bugs |
| `sybil/datasets/nlst_risk_factors.py` | `NLSTRiskFactorVectorizer` (11 keys) | Off by default |
| `sybil/datasets/utils.py` | `VOXEL_SPACING`, `CENSORING_DIST`, `order_slices`, mask scaling | Core |
| `sybil/utils/device_utils.py` | GPU/CPU/MPS selection | Core |
| `sybil/utils/loading.py` | `get_sample_loader`, custom collate | Training |
| `sybil/utils/losses.py` | Survival + annotation losses | 3 of 6 functions unused |
| `sybil/utils/metrics.py` | Per-year AUC, Uno's c-index, KM censoring dist | Core |
| `sybil/utils/sampler.py` | `DistributedWeightedSampler` | Training |
| `sybil/utils/visualization.py` | Attention heatmap GIFs | Hardcoded geometry |
| `sybil/utils/logging_utils.py` | Logger setup | — |
| `scripts/train.py` | `SybilLightning` training module | Training entry point |
| `scripts/evaluate.py` | Evaluation script | — |
| `scripts/plcom2012/` | PLCOm2012 clinical baseline | Paper comparator |
| `scripts/data/` | NLST metadata JSON + MD.ai annotation parsing | Hardcoded MIT paths |
| `files/*.csv` | Empty schema templates | Reference only; no consumers |
| `examples/local.py`, `examples/remote_ark.py` | Local and Ark-server usage | — |
| `tests/` | 1 real unit test; regression tests env-gated | Low default coverage |
