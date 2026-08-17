# DINOv2 Domain Response Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract frozen DINOv2 patch tokens, align them with COCO domain masks, quantify domain-area association, and render paper-ready spatial response figures.

**Architecture:** Add a focused `domain_response.py` module containing pure mask/alignment/statistics functions plus a CLI runner. Keep the existing global-embedding classifier unchanged. A Slurm launcher supplies the shared offline DINOv2 cache and writes all artifacts to a new supplement directory.

**Tech Stack:** Python 3.10, NumPy, Pillow, timm, PyTorch, scikit-learn, SciPy, Matplotlib, COCO polygon JSON, Slurm.

---

### Task 1: Mask rasterization and preprocessing alignment

**Files:**
- Create: `src/rate_identification/domain_response.py`
- Create: `tests/test_domain_response.py`

- [ ] **Step 1: Write failing tests for polygon union and deterministic crop alignment**

```python
def test_rasterize_union_does_not_double_count_overlap():
    mask = rasterize_polygon_union((20, 10), [[[0, 0, 10, 0, 10, 10, 0, 10]], [[5, 0, 15, 0, 15, 10, 5, 10]]])
    assert mask.shape == (10, 20)
    assert int(mask.sum()) == 160

def test_align_image_and_mask_returns_518_square_pair():
    image_out, mask_out = align_image_and_mask(image, mask, resize_long_edge=1024, input_size=518)
    assert image_out.size == (518, 518)
    assert mask_out.shape == (518, 518)
```

- [ ] **Step 2: Run `python -m unittest tests.test_domain_response -v` and confirm missing-function failures**

- [ ] **Step 3: Implement polygon union and identical resize/center-crop geometry**

```python
def rasterize_polygon_union(size, segmentations):
    canvas = Image.new("L", size, 0)
    draw = ImageDraw.Draw(canvas)
    for polygons in segmentations:
        for polygon in polygons:
            if len(polygon) >= 6:
                draw.polygon(list(zip(polygon[0::2], polygon[1::2])), fill=1)
    return np.asarray(canvas, dtype=np.uint8)

def patch_occupancy(mask, patch_size=14):
    grid = mask.shape[0] // patch_size
    return mask.reshape(grid, patch_size, grid, patch_size).mean(axis=(1, 3))
```

- [ ] **Step 4: Run the mask tests and confirm they pass**

### Task 2: Prototype response and threshold selection

**Files:**
- Modify: `src/rate_identification/domain_response.py`
- Modify: `tests/test_domain_response.py`

- [ ] **Step 1: Write failing synthetic-token tests**

```python
def test_prototype_response_is_higher_for_domain_tokens():
    domain, background = fit_weighted_prototypes(tokens, occupancy)
    scores = prototype_response(tokens, domain, background)
    assert scores[0] > scores[-1]

def test_threshold_selection_maximizes_validation_dice():
    threshold, metrics = select_threshold(scores, occupancy)
    assert metrics["dice"] == 1.0
```

- [ ] **Step 2: Verify the new tests fail for missing behavior**

- [ ] **Step 3: Implement L2-normalized weighted prototypes, cosine-difference scores, threshold search, Dice/IoU and area fractions using `fit_weighted_prototypes(tokens, occupancy)`, `prototype_response(tokens, domain, background)` and `select_threshold(scores, occupancy)`**

- [ ] **Step 4: Run all unit tests**

### Task 3: Frozen DINOv2 patch extraction and COCO split runner

**Files:**
- Modify: `src/rate_identification/domain_response.py`

- [ ] **Step 1: Add a testable `split_patch_tokens` helper and verify prefix-token removal on a synthetic tensor**

- [ ] **Step 2: Implement CLI arguments `--data-root`, `--output-dir`, `--resize-long-edge`, `--input-size`, `--patch-size`, `--seed` and `--reuse-cache`**

- [ ] **Step 3: Load `vit_small_patch14_dinov2` offline, call `forward_features`, remove the CLS token, and assert `(1369, 384)` per image**

- [ ] **Step 4: Save float16 token caches and aligned occupancy arrays with source metadata**

### Task 4: Statistics and paper-ready figures

**Files:**
- Modify: `src/rate_identification/domain_response.py`
- Modify: `tests/test_domain_response.py`

- [ ] **Step 1: Write failing tests for per-image table rows and within-magnification centering**

- [ ] **Step 2: Implement Test Pearson/Spearman, within-class centered correlation, patch AUROC, Dice, IoU and inside/outside response summaries**

- [ ] **Step 3: Generate representative original/mask/heatmap/overlay panels, area scatter, response distribution, CSV/JSON/SVG/PNG and manifest**

- [ ] **Step 4: Run all unit tests and inspect two rendered figures**

### Task 5: Slurm execution and verification

**Files:**
- Create: `scripts/run_dino_domain_response.sh`
- Modify: `README.md`

- [ ] **Step 1: Add the offline `HF_HOME`, `HF_HUB_OFFLINE`, writable `TMPDIR`, Conda and output configuration**

- [ ] **Step 2: Validate with `bash -n scripts/run_dino_domain_response.sh`**

- [ ] **Step 3: Commit and push the experiment branch**

- [ ] **Step 4: Submit the GPU job and record its Slurm ID**

- [ ] **Step 5: Verify exit code 0, empty stderr, expected token shapes, all plots, CSV row counts and statistics before reporting any scientific conclusion**
