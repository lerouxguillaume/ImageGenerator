# Auto Generate Candidate Pipeline

## Goal

Let the user describe an asset requirement, then have the system generate and refine candidates until it can propose a small set of usable assets.

Status: first implementation slice is active for `Wall Left` under the existing `GenerationWorkflow::PhasedRefinement` enum. The current code writes `runs/<run_id>/explore/`, `runs/<run_id>/refine/`, and `manifest.json`.

The user should be able to provide:

- project theme, optional
- asset name
- asset detail prompt, for example `stone wall with small arched window`

The system should own the production constraints:

- left wall orientation
- single isolated object
- transparent background
- no floor plane
- no room scene
- expected bounds, anchor, and fill ratio

## Pipeline

Replace the current linear phase model with a candidate-run pipeline:

```text
Exploration batch
-> correctness scoring
-> candidate pool
-> refinement batch
-> final scoring
-> proposed assets
```

## Exploration Phase

Generate a fixed batch, probably `8` images by default.

Score each raw image for correctness:

- background can be removed
- occupied bounds are close enough
- anchor is close enough
- fill ratio is acceptable
- image is not empty
- image is not a full-frame scene
- no obvious floor or room leakage, where detectable

Output expectations:

- all generated images remain available for debugging
- top `3` candidates enter the refinement pool

## Candidate Pool

Store a lightweight run manifest instead of relying only on directory names:

```json
{
  "runId": "...",
  "assetTypeId": "...",
  "exploration": [],
  "candidates": [
    {
      "rawPath": "...",
      "processedPath": "...",
      "correctnessScore": 123,
      "status": "candidate"
    }
  ]
}
```

This lets the UI explain what happened and makes later scoring/ranking easier to debug.

## Refinement Phase

For each top candidate:

- run img2img from the raw candidate
- use moderate denoise, for example `0.28`
- generate `2` variants per candidate

That gives:

```text
3 candidates x 2 refinements = 6 refined images
```

Re-score refined outputs for correctness.

## Final Proposal Phase

Sort refined images by correctness score.

Mark results as:

- `BEST` - lowest score
- `OK` - below threshold
- `NEAR` - close but imperfect
- `REJECTED` - structurally bad

Default gallery view should show proposed assets first, not every raw attempt.

## UI Changes

Replace phase-focused controls with run-focused language:

- button: `Generate Candidates`
- status: `Exploring`, `Refining`, `Scoring`, `Done`
- gallery badges: `BEST`, `OK`, `NEAR`
- optional debug toggle: `All Attempts`

Manual `Refine Best` should be hidden for the MVP unless needed for debugging.

## Scoring

Keep correctness scoring deterministic first.

Immediate scoring fields:

- canvas mismatch
- alpha or removable background
- empty subject
- fill ratio
- bounds error
- anchor error

Later scoring fields:

- duplicate or similarity penalty
- texture/detail quality
- prompt/detail adherence
- LLM or vision-model review

## Stop Conditions

Do not generate forever.

MVP limits:

- exploration: `8` images
- candidate pool: top `3`
- refinement: `2` variants per candidate
- max total generated: `14`

Stop early only if enough `OK` candidates exist, for example `3`.

## Implementation Order

1. Rename concepts internally from phase to `CandidateRun` / `Attempt`.
2. Add run manifest JSON.
3. Generate exploration batch into `runs/<run_id>/explore/`.
4. Score exploration and select top `3`.
5. Generate refinements into `runs/<run_id>/refine/`.
6. Score refinements and sort gallery by status/score.
7. Update UI labels and badges.
8. Keep old phase directories readable only if needed for migration/debugging.

## Bad Ideas To Avoid

- Infinite generation until success.
- Refining every image.
- Refining images that fail basic structure.
- Letting user prompt override geometry constraints.
- Building support for more asset types before this loop works for walls.
- Using processed 128x192 exports as the only scoring source.
- Hiding failed attempts completely; keep them available for debugging.
