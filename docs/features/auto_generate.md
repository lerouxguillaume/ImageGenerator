# Auto Generate Candidate Pipeline

## Goal

Let the user describe an asset requirement, then have the system generate and refine candidates until it can propose a small set of usable assets.

Status: first implementation slice is active for `Wall Left` under `GenerationWorkflow::CandidateRun`.

The user provides:

- project theme (optional)
- asset name
- asset detail prompt, e.g. `stone wall with small arched window`

The system owns the production constraints:

- left wall orientation
- single isolated object
- transparent background
- no floor plane
- no room scene
- expected bounds, anchor, and fill ratio

---

## Pipeline

```text
Patron generation (per asset type)
-> Exploration batch (patron-seeded img2img)
-> Correctness scoring
-> Candidate pool
-> Refinement batch (candidate-seeded img2img)
-> Final scoring
-> Proposed assets
```

---

## Patron Phase

Before any generation, the system produces a **patron** — a shape-correct reference image derived from the asset type's `AssetSpec`. The patron defines the silhouette contract for the exploration batch.

**Patron lifecycle:**
- Generated once when an asset type is created from a template
- Regenerated automatically when the user changes orientation or expected bounds in the UI
- Stored at `output/<project>/<asset>/patron.png` (one per asset type, shared across all runs)
- Loaded lazily at run start if the file is missing

**Patron geometry (by orientation):**

| Orientation | Shape |
|---|---|
| `LeftWall` | Isometric parallelogram: skew = bounds.w / 4, top-right corner highest |
| Other | Bounding box rectangle (fallback until orientation-specific shapes are added — see backlog M4) |

Fill: neutral mid-gray RGB(128, 128, 128) on a fully transparent canvas. The patron is intentionally flat and textureless — it anchors the SD model to the correct position and aspect ratio without prescribing surface detail.

**Source:** `src/projects/PatronGenerator.cpp`

---

## Exploration Phase

Generate a fixed batch (default: 8 images) seeded from the patron via img2img.

- `initImagePath` = `output/<project>/<asset>/patron.png`
- `strength` = 0.70 (patron provides shape anchor; model has 70% freedom for texture/material)
- Falls back to txt2img if the patron file is missing or write failed

Score each raw image for correctness:

- background can be removed
- occupied bounds are close enough
- anchor is close enough
- fill ratio is acceptable
- image is not empty
- image is not a full-frame scene

All generated images remain available in `explore/raw/` and `explore/processed/` for debugging.

Top 3 candidates enter the refinement pool.

---

## Candidate Pool

A run manifest is written to `runs/<run_id>/manifest.json`:

```json
{
  "runId": "...",
  "assetTypeId": "...",
  "exploration": [],
  "proposals": [
    {
      "rawPath": "...",
      "processedPath": "...",
      "correctnessScore": 123,
      "status": "best"
    }
  ]
}
```

Lower `correctnessScore` is better.

---

## Refinement Phase

For each top exploration candidate:

- run img2img from the candidate's raw output
- strength: `refinementStrength` (default 0.27)
- generate 2 variants per candidate

```text
3 candidates × 2 refinements = 6 refined images
```

Re-score refined outputs for correctness.

---

## Final Proposal Phase

Sort refined images by correctness score.

Status labels:

| Label | Meaning |
|---|---|
| `best` | Lowest score overall |
| `near` | Below threshold but not the best |
| `rejected` | Structurally bad (score exceeds threshold) |

Default gallery view shows proposals first. Failed explore attempts are visible in `explore/processed/` for debugging.

---

## Output Layout

```text
output/<project>/<asset>/
    patron.png                      ← shared across all runs
    runs/
        run_<id>/
            explore/
                raw/
                processed/
            refine/
                raw/
                processed/
            manifest.json
```

---

## Scoring

Correctness scoring is deterministic and geometric. Lower is better.

Current scoring fields:

| Field | Penalty |
|---|---|
| Canvas size mismatch | large fixed penalty |
| No removable background | large fixed penalty |
| Fill ratio out of range | proportional to deviation from target |
| Bounds error | proportional to deviation from expectedBounds, capped |
| Anchor error | proportional to deviation from spec anchor |

Later scoring improvements (see backlog C2):

- Parameterise as `ScoringPolicy` so non-wall assets can have different weights
- Silhouette match against patron mask
- Duplicate or similarity penalty
- Prompt/detail adherence via vision model

---

## Generation Settings (CandidateRunSettings)

Defaults hardcoded in `ImageGeneratorController::CandidateRunSettings`. Moving these to `AssetType` for per-asset configurability is tracked in backlog item M3.

| Setting | Default | Meaning |
|---|---|---|
| `minExploreImages` | 8 | Exploration batch size |
| `candidateCount` | 3 | Top candidates entering refinement |
| `refineVariants` | 2 | Refinement variants per candidate |
| `explorationStrength` | 0.70 | Img2img denoise for exploration |
| `refinementStrength` | 0.27 | Img2img denoise for refinement |
| `scoreThreshold` | 150.0 | Score above which a proposal is `rejected` |

---

## Stop Conditions

- Exploration: fixed at `minExploreImages`
- Refinement: fixed at `candidateCount × refineVariants`
- No infinite generation loop

---

## Bad Ideas To Avoid

- Infinite generation until success.
- Refining every image.
- Refining images that fail basic structure.
- Letting user prompt override geometry constraints.
- Building support for more asset types before this loop works for walls.
- Using processed 128×192 exports as the only scoring source.
- Hiding failed attempts completely; keep them in `explore/` for debugging.
