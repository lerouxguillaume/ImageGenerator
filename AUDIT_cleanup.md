# ImageGenerator — Dead-Code & Cleanup Audit

Five parallel investigations (filesystem-discovery remnants, constants↔Theme, orphan files/game code, controller/docs quality, CMake) plus independent spot-verification. **Headline:** the two named removals (filesystem discovery, Projects/asset-pack) were done *cleanly at the mechanism level* — there is no live second discovery path, and `docs/` is already consistent. What remains is (a) vestigial fields/branches the removals left dangling, (b) a large island of **pre-existing game-era dead code** the removals exposed, and (c) the still-incomplete `constants.hpp`→`Theme` migration.

Nothing below has been changed.

---

## 1. Safe to delete now (high confidence)

### 1a. Orphan files — the `src/ui/portraits/` island + dead widget base
Verified zero external includers and zero symbol references (grep-confirmed, not just the GLOB file scan):

| File(s) | Evidence |
|---|---|
| `src/ui/portraits/PortraitGenerator.{hpp,cpp}` | `PortraitGenerator` referenced only in its own two files. Procedural (non-AI) game portrait drawer. |
| `src/ui/portraits/PortraitLayer.{hpp,cpp}` | `PortraitLayer` used only inside the above (itself orphaned). Dead transitively. |
| `src/portraits/PromptBuilder.hpp` | Included only by `PortraitGeneratorAi.cpp`; its `buildCharacterPrompt`/`buildNegativePrompt` are used only by the never-called `generatePortrait()` (see 1b). |
| `src/ui/widgets/base/WidgetController.h` | 0 includers; `WidgetController` never inherited/referenced anywhere. |
| `src/ui/widgets/base/WidgetEvent.h` | Included only by orphaned `WidgetController.h`; `WidgetEvent` + all its enum values unreferenced. |

Keep (verified live, do **not** delete): `src/ui/widgets/base/Widget.h` (`MultiLineTextArea : public Widget`), `src/portraits/ClipTokenizer.*` (used by `SdPipeline`/`SdTextEncoder`), `PortraitGeneratorAi.*` (its `generateFromPrompt` is the live entry at `GenerationService.cpp:28`, and the header hosts `GenerationParams`/`LoraEntry`).

> After deleting any `.cpp`, re-run `cmake -B build` (GLOB_RECURSE).

### 1b. Dead functions inside live files
- `PortraitGeneratorAi::generatePortrait(...)` — decl `PortraitGeneratorAi.hpp:35`, def `.cpp:7`. Never called (only a comment mention at `SdPipeline.hpp:13`). Sole consumer of `PromptBuilder.hpp` + the `Race`/`Gender` enums.
- `Helpers::statusStr(Status)` / `Helpers::statusColor(Status)` — declared `Helpers.hpp:26-27`, **never defined**, never called.
- `Helpers::diffStars(int)` — declared `Helpers.hpp:25`, **never defined**, never called.
- `Helpers::diffColor(int)` — defined `Helpers.cpp:100`, never called. (This is the *only* reader of `colors().purpleLt`, so that Theme field dies with it.)

### 1c. Dead enum variants — `src/enum/enums.hpp`
Zero references anywhere (High): `AdventurerClass` (l.27), `QuestType` (l.37), `Phase` (l.40), `Role` (l.42), `PartyRole` (l.45), `Status` (l.35, the game `{Idle,Assigned,Injured}` — referenced only by the dead `Helpers::statusStr/statusColor` decls).
All 7 `GenerationStage` variants are emitted — none dead. `ModelType` live. (`Race`/`Gender` → see §2, they fall once the portrait chain in 1a/1b is gone.)

### 1d. Dead `constants.hpp` color/layout aliases (zero users)
Pure aliases with no call sites — deletable from `constants.hpp` with no migration:
- Colors: `Col::Overlay`, `Col::Red`, `Col::RedLt`, `Col::Green`, `Col::GreenLt`, `Col::Purple`, `Col::PurpleLt`, `Col::Injury`.
- Layout: `BODY_H_FULL`, `BODY_H_LLM` (derived, 0 users).
- Dead include: `src/views/MenuView.cpp:4` `#include "../enum/constants.hpp"` — file uses no `Col::`/`Ui` symbol (already on `Theme::instance()`).

> Nuance: `Col::Injury` the *alias* is dead, but `UiColors::injury` in `Theme.h` is **live** (`ImportModelModal.cpp:241`, Warn badge). Delete the alias, keep the Theme field.

### 1e. Dead ResultPanel fields + their controller writers/handlers (removed-subsystem residue)
All confirmed set-only-to-off / never-truthy:
- `ResultPanel.hpp:68-70` `selectedReferenceUsed` / `selectedReferenceImage` / `selectedStructureStrength` — only ever written `false`/`.clear()`/`0.0f` (controller `:218-220`, `:471-473`). Render branch `ResultPanel.cpp:225-233` is permanently dead.
- `ResultPanel.hpp:65,67` `showOutputModeToggle` (never set `true`) + `outputModeChanged`. Render `ResultPanel.cpp:160,208`, toggle rects `:408,413`, and controller handler `:607-610` are all unreachable.
- `ResultPanel.hpp:77-78` `bestWallCandidateScore` — only ever set to `-1.f` (controller `:492`); score badge `ResultPanel.cpp:346-351` never renders.
- `ResultPanel.hpp:23-26` `struct GalleryTab` + `:36 showTabs` / `:39-41 tabs`/`activeTabIndex`/`tabChanged` — `tabs` is never populated anywhere, so all tab render/click code (`ResultPanel.cpp:104,296,327,444`) is dead.

Deleting each field lets you also drop its controller writer and dead render branch in the same pass.

### 1f. Dead struct field
- `GenerationJob::vaeEncoderAvailable` (`GenerationService.hpp:18`) — **write-only**: set at `ImageGeneratorController.cpp:381`, never read by `GenerationService::run` (pipeline derives VAE-encoder availability independently via `SdLoader`→`ctx.vaeEncoderAvailable`). (Touches a CLAUDE.md "Key fact" — see §4.)

### 1g. Unused includes
- `src/ui/widgets/ImportModelModal.hpp:2` `#include <filesystem>` — no filesystem use (discovery-removal residue).
- `src/views/MenuView.cpp:3` `#include <filesystem>` — unused.
- `src/enum/enums.hpp:3` `#include <string>` — file is enums only (Medium).

### 1h. Dead CMake step
- `CMakeLists.txt:185-190` — copies `assets/references/`, which exists but is **empty** and has zero references in `src/`/`docs/`; it was tied to the removed Projects/asset subsystem. Dead relic.

---

## 2. Probably removable — verify first

| Item | What to verify |
|---|---|
| Enums `Race` (l.30) / `Gender` (l.32) | Only used by the dead portrait chain (1a/1b). Remove **after** deleting `PortraitGenerator*`/`PromptBuilder.hpp`/`generatePortrait`. |
| Scan-era `path.filename()` display-name fallbacks — `ImageGeneratorController.cpp:107-109, 229-231, 811-813`; `SettingsPanel.cpp:110-114, 334-336` | These `... : path(...).filename()` else-branches are unreachable because the population loop always fills `availableModelNames` whenever `availableModels` is non-empty. Confirm the empty-list path before pruning; they also violate the CLAUDE.md:157 invariant. |
| `modelConfigs` keyed by raw `filename()` — `ImageGeneratorController.cpp:350, 427-432` | Inconsistent with `previewKey` (`:811`) which keys by registry `name`. Decide the canonical key (folder vs registry name) and unify; update `AppConfig.hpp:41` comment. |
| `"models"` sentinel — `SettingsPanel.cpp:19-20` (`getSelectedModelDir` returns `"models"` when empty) + comment `hpp:71`, controller comment `:691-692` | Under registry-only discovery there is no scan `models/` dir to load; this returns a non-loadable path. Consider returning empty + explicit no-model handling. |
| `CMakeLists.txt:228-236` `assets/fonts/arial.ttf` copy | Source file is **missing**; `EXISTS`-guarded so it silently no-ops (Windows falls back to `C:/Windows/Fonts`). Decide: restore the asset or delete the step. |
| `ImportedModel::importedAt` (`ImportedModelRegistry.hpp:16`) | JSON round-trips but never consumed. Low priority — keep if you want the metadata, else drop from struct + JSON. |
| Theme dead palette — `UiColors::{purple, purpleLt, buttonDisabledBg, shadow, surfaceRaised, surfaceInset, accentSoft}` in `Theme.h` | Grep for direct `colors().<x>` use; `purpleLt` only via dead `diffColor`. Prune once §1b lands. Keep `injury` (live). |
| `MultiLineTextArea.hpp:54` `textColor_ = sf::Color(220,210,185) // Col::Text default` | Stale literal — no longer equals `UiColors::text` {232,238,245}. Reconcile to the Theme value. |

---

## 3. Refactoring opportunities

**R1 — Collapse the 5 parallel model vectors into one struct.** *(Medium effort, high benefit)*
`SettingsPanel.hpp:24-28` (`availableModels`/`availableModelNames`/`availableModelTypes`/`modelVaeEncoderAvailable`/`modelLoraCompatible`) are filled 1:1 from a single `ImportedModel` in one loop (`ImageGeneratorController.cpp:693-711`). Now that discovery is single-source, replace with `std::vector<ModelEntry>` (or reuse `ImportedModel` + a derived `ModelType`). Eliminates the parallel-length invariant, five `clear()`s, and five `[idx]` bounds-guards; makes `currentModelVaeEncoderAvailable()`/`currentModelLoraCompatible()`/`currentModelType()` a single `idx < entries.size()` check. This is the single highest-value cleanup.

**R2 — Finish the `constants.hpp` → `Theme` migration and retire the file.** *(Medium effort, medium benefit)*
After §1d, the remaining `Col::*` aliases (`Bg, Panel, Panel2, Border, BorderHi, Gold, GoldLt, Text, Muted, Blue, BlueLt`, ~109 sites; `Muted`/`GoldLt`/`Panel2` heaviest) and metric aliases (`WIN_W, WIN_H, HEADER_H, LEFT_W, PAD, MENU_BAR_H, LLM_BAR_H, LLM_EXPANDED_H, LEFT_PANEL_W`) should be migrated to `Theme::instance().colors().<x>` / a `Ui`-metrics accessor. Five composites have no `UiMetrics` field (`MID_X, MID_W, BODY_H, BODY_Y`, plus `LOG_H` which only feeds `BODY_H`) — either recompute at call sites or add them to Theme. Once done, `constants.hpp` disappears.

**R3 — Trim controller cruft freed by the removals.** *(Low effort, low-medium benefit)*
- `saveSettings` (`ImageGeneratorController.cpp:292-295`) clears only 3 of 5 model vectors then sets `modelsDirty=true`, which re-clears all 5 — the three manual `clear()`s are redundant and misleading; delete them (keep `modelsDirty=true`).
- Trivial forwarders `triggerGeneration`/`openSettingsDialog` (`:823-829`) — inline or promote `launchGeneration`/`openSettings` to the public entry points.
- Over-engineered delete path (`:630-654`, `relative()` + `..`-traversal, comment `:644 "project/assettype/"`) was built for the removed project subdir layout; the gallery is now a flat `outputDir` scan — a "direct child of outputDir" check suffices.

**R4 — Make a CLAUDE.md invariant enforceable by construction.** *(Low effort, low benefit)*
Once R1 lands, "never derive display name from `path.filename()`" (CLAUDE.md:157) becomes structurally impossible (name comes from `ImportedModel::name`), and the `job.vaeEncoderAvailable`/`loraCompatible` capability flags can flow `ImportedModel::capabilities` → job directly rather than through the panel — reducing convention-only invariants.

**R5 — Naming.** Rename the "model scan" wording on registry-era code: `architecture_summary.md:30` "model/LoRA scan", controller comments `:691-692`. Models are read from the registry, not scanned.

---

## 4. Docs & CLAUDE.md to update

The **`docs/` tree and CLAUDE.md are already consistent** with filesystem-discovery removal, the merged generate/edit screen, and `constants.hpp`-as-glue. Only these need touching:

**CLAUDE.md**
- "Generation service — Key facts": `GenerationJob carries … model capabilities (vaeEncoderAvailable, loraCompatible)` — drop `vaeEncoderAvailable` if §1f is removed.
- Invariant `:157` ("Never derive model display name from `path.filename()`") — becomes enforced-by-construction after R1; note the scan-era fallbacks in §2 still violate it today.

**docs/00_overview/build_system.md** (materially out of date)
- L3 wrong project name ("**GuildMaster**" → ImageGenerator).
- Missing the real `USE_GENAI` option and `ONNXRUNTIME_ROOT`/`ORT_GENAI_ROOT` path vars.
- Omits all POST_BUILD copy steps (config, models, scripts, generated dir, Windows DLL/font).
- L39-47 GLOB guidance: CMakeLists uses `CONFIGURE_DEPENDS` (auto-reglob on build), so the "must manually re-run cmake" wording is overstated (keep as a belt-and-suspenders caveat, but note it).

**docs/80_presets/preset_overview.md**
- `:129` "linear scan of `panel.availableModels` **by folder name**" and `:145` sample `gs.modelId = filesystem::path(...).filename()` — update to registry `name`-based matching (verify against `PresetManager.cpp`).

**docs/00_overview/architecture_summary.md**
- `:30` "model/LoRA **scan**" → "model registry read / LoRA scan".

**Root (not under docs/, but stale):**
- `README.md` (l.3,14,30-31,52,165) still describes Project workspace / asset packs / candidate-run / `src/assets/` / `src/projects/` / `ProjectManager` — all removed.
- `backlog.md` references removed symbols (`AssetArtifactStore`, `CandidateRunPipeline`, `ProjectManager`, `AssetType`, …). Confirm whether it's an intentional historical log before editing.
- Stale comments: `CMakeLists.txt:166` "portrait models (ONNX + tokenizer)" (only tokenizer files present); `ImageGeneratorController.cpp:644` "project/assettype/".

---

## Suggested deletion order (respects dependencies)
1. §1a orphan files → then §2 `Race`/`Gender` enums fall.
2. §1b `generatePortrait` + `Helpers::*` → then `PromptBuilder.hpp` and Theme `purpleLt` fall.
3. §1c–1g in any order; §1e best done field-by-field with its controller writer + render branch together.
4. `cmake -B build` after every `.cpp` removal, then `cmake --build build`, then run the binary (no tests) to confirm.
