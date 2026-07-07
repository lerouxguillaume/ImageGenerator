#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Golden-run regression harness for the SD pipeline.
#
# This is the regression net for the Phase 1 pipeline refactor (and the feature
# phases after it). It drives the app's --headless-generate mode with fixed,
# scripted parameters and hash-compares the PNG outputs of two builds.
#
# ONE-COMMAND USAGE
#   Capture goldens from the CURRENT (pre-refactor) build:
#     scripts/golden/golden_harness.sh capture
#   Verify a new build produces bit-identical output:
#     scripts/golden/golden_harness.sh verify
#   Check whether this machine is bit-deterministic at all (run first!):
#     scripts/golden/golden_harness.sh determinism
#
# The combos below cover txt2img (×2), a multi-image batch (seed-sequence), and
# img2img (the init-latent path being migrated). Goldens live in
# scripts/golden/goldens/ and are meant to be committed alongside this script.
#
# IMPORTANT: goldens must be captured from a build made BEFORE the Step 2
# refactor. Git makes this easy — capture at the harness commit, then verify
# after the refactor commit.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
GOLDENS="$HERE/goldens"
COMPARE="$HERE/compare.py"

# ── Defaults (override with flags) ───────────────────────────────────────────
MODE_CMD="${1:-}"; shift || true
BIN="$REPO/build/image_generator"
MODEL=""                 # auto-detected from registry.json if empty
CMP_MODE="exact"         # exact | tolerance
TOL=2                    # tolerance mode: max per-channel abs diff
MAX_FRAC=0.001           # tolerance mode: max fraction of pixels over TOL

while [ $# -gt 0 ]; do
    case "$1" in
        --bin)       BIN="$2"; shift 2 ;;
        --model)     MODEL="$2"; shift 2 ;;
        --mode)      CMP_MODE="$2"; shift 2 ;;
        --tol)       TOL="$2"; shift 2 ;;
        --max-frac)  MAX_FRAC="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

[ -x "$BIN" ] || { echo "ERROR: binary not found/executable: $BIN" >&2; exit 2; }

# Auto-detect the first imported model from the registry when --model is absent.
if [ -z "$MODEL" ]; then
    REG="$REPO/models/imported/registry.json"
    if command -v python3 >/dev/null 2>&1 && [ -f "$REG" ]; then
        MODEL="$(python3 -c "import json,sys; m=json.load(open(sys.argv[1]))['models']; print(m[0]['onnxPath'] if m else '')" "$REG" || true)"
    fi
fi
[ -n "$MODEL" ] || { echo "ERROR: no --model given and none auto-detected from registry.json" >&2; exit 2; }
MODEL="$(cd "$MODEL" && pwd)"   # absolute — the binary chdir's to its own dir
echo "model : $MODEL"
echo "bin   : $BIN"

# ── Combo definitions ────────────────────────────────────────────────────────
# run_combo <name> <extra args...>  → writes $WORK/<name>.png (or _N.png for batches)
run_combo() {
    local name="$1"; shift
    echo "  [$name] $*"
    "$BIN" --headless-generate --model "$MODEL" --out "$WORK/$name.png" "$@" \
        >>"$WORK/harness.log" 2>&1
}

run_all() {
    run_combo c1 --prompt "a photorealistic portrait of a red fox in a forest" \
                 --neg "blurry, lowres, deformed" --seed 42 --steps 12 --guidance 7.5
    run_combo c2 --prompt "a serene mountain lake at sunrise, mist" \
                 --neg "" --seed 7 --steps 12 --guidance 8.0
    run_combo c3 --prompt "a still life of fruit on a wooden table" \
                 --neg "" --seed 100 --steps 8 --images 2 --guidance 7.0
    # img2img: init is c1's freshly-generated output, so the whole chain is
    # reproduced from scratch each run (exercises encodeImage + startStep).
    run_combo c4 --prompt "a photorealistic portrait of a red fox, autumn palette" \
                 --neg "blurry, lowres, deformed" --seed 42 --steps 12 --guidance 7.5 \
                 --init "$WORK/c1.png" --strength 0.5
}

# Files each run_all is expected to produce (c3 is a 2-image batch → _1/_2).
EXPECTED=(c1.png c2.png c3_1.png c3_2.png c4.png)

MANIFEST="$GOLDENS/hashes.txt"

sha_of() { sha256sum "$1" | cut -d' ' -f1; }

# sha256 recorded for basename $1 in the committed manifest, or "" if absent.
manifest_hash() {
    [ -f "$MANIFEST" ] || { echo ""; return; }
    awk -v f="$1" '$2==f {print $1}' "$MANIFEST" | head -n1
}

compare_one() {
    local golden="$1" cand="$2"
    if [ "$CMP_MODE" = "exact" ]; then
        if cmp -s "$golden" "$cand"; then echo "PASS(exact)"; return 0; else echo "FAIL(exact)"; return 1; fi
    else
        python3 "$COMPARE" "$golden" "$cand" --tolerance "$TOL" --max-frac "$MAX_FRAC"
    fi
}

# Verify one produced file against the committed goldens. Prefers a byte/tolerance
# compare against a golden PNG when present; otherwise falls back to the committed
# sha256 manifest (the cross-machine path — PNGs are gitignored, hashes.txt is not).
verify_one() {
    local name="$1" cand="$2"
    if [ -f "$GOLDENS/$name" ]; then
        printf "  %-14s " "$name"; compare_one "$GOLDENS/$name" "$cand"; return $?
    fi
    local want; want="$(manifest_hash "$name")"
    if [ -n "$want" ]; then
        printf "  %-14s " "$name"
        if [ "$(sha_of "$cand")" = "$want" ]; then echo "PASS(hash)"; return 0
        else echo "FAIL(hash)"; return 1; fi
    fi
    echo "  $name: NO GOLDEN (neither PNG nor hashes.txt entry)" >&2
    return 1
}

case "$MODE_CMD" in
    determinism)
        WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
        echo "== determinism check (same inputs, twice) =="
        run_combo c1  --prompt "a photorealistic portrait of a red fox in a forest" \
                      --neg "blurry, lowres, deformed" --seed 42 --steps 12 --guidance 7.5
        run_combo c1b --prompt "a photorealistic portrait of a red fox in a forest" \
                      --neg "blurry, lowres, deformed" --seed 42 --steps 12 --guidance 7.5
        if cmp -s "$WORK/c1.png" "$WORK/c1b.png"; then
            echo "RESULT: DETERMINISTIC — use 'capture'/'verify' in exact mode (default)."
        else
            echo "RESULT: NON-DETERMINISTIC on this EP — two identical runs differ."
            echo "        Use tolerance mode:  verify --mode tolerance --tol $TOL --max-frac $MAX_FRAC"
            echo "        or rebuild CPU-only (deterministic) for the equivalence check."
        fi
        ;;
    capture)
        mkdir -p "$GOLDENS"; WORK="$GOLDENS"; : >"$WORK/harness.log"
        echo "== capturing goldens into $GOLDENS =="
        run_all
        # Write a sha256 manifest next to the PNGs. Commit hashes.txt (not the
        # PNGs, which are gitignored) so goldens are verifiable cross-machine.
        : >"$MANIFEST"
        echo "captured:"; for f in "${EXPECTED[@]}"; do
            [ -f "$GOLDENS/$f" ] || { echo "  MISSING $f" >&2; exit 1; }
            printf "%s  %s\n" "$(sha_of "$GOLDENS/$f")" "$f" >>"$MANIFEST"
            echo "  $f  $(sha_of "$GOLDENS/$f" | cut -c1-16)"
        done
        echo "manifest: $MANIFEST"
        ;;
    verify)
        [ -f "$MANIFEST" ] || [ -d "$GOLDENS" ] || { echo "ERROR: no goldens; run 'capture' first" >&2; exit 2; }
        WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
        echo "== verifying against $GOLDENS (mode=$CMP_MODE) =="
        run_all
        fails=0
        for f in "${EXPECTED[@]}"; do
            [ -f "$WORK/$f" ] || { echo "  $f: NOT PRODUCED" >&2; fails=$((fails+1)); continue; }
            verify_one "$f" "$WORK/$f" || fails=$((fails+1))
        done
        echo "-----"
        if [ "$fails" -eq 0 ]; then echo "ALL GOLDENS MATCH ✅"; else echo "$fails MISMATCH(ES) ❌"; exit 1; fi
        ;;
    hires-repro)
        # Reproducibility: same seed + hires ON, twice → identical bytes.
        WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
        echo "== hires reproducibility (same seed + hires, twice) =="
        run_combo h1 --prompt "a photorealistic portrait of a red fox in a forest" \
                     --neg "blurry, lowres, deformed" --seed 42 --steps 12 --guidance 7.5 \
                     --hires --hires-scale 1.5 --hires-strength 0.5
        run_combo h2 --prompt "a photorealistic portrait of a red fox in a forest" \
                     --neg "blurry, lowres, deformed" --seed 42 --steps 12 --guidance 7.5 \
                     --hires --hires-scale 1.5 --hires-strength 0.5
        if cmp -s "$WORK/h1.png" "$WORK/h2.png"; then
            echo "RESULT: hires is REPRODUCIBLE ✅ ($(sha_of "$WORK/h1.png" | cut -c1-16))"
        else
            echo "RESULT: hires NON-REPRODUCIBLE ❌ — same seed produced different output" >&2; exit 1
        fi
        ;;
    cancel-restore)
        # Cancel mid-hires-pass, then generate native in the SAME process and
        # compare against the c1 golden. Proves ScopedLatentResolution restored
        # ctx dims on the cancel path. --steps 12 → base is 12; cancel at 15
        # lands inside pass 2. --out receives the follow-up native generation.
        [ -f "$MANIFEST" ] || [ -f "$GOLDENS/c1.png" ] || { echo "ERROR: capture c1 golden first" >&2; exit 2; }
        WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
        echo "== cancel-restore (cancel mid-hires → native must match c1 golden) =="
        run_combo restore --prompt "a photorealistic portrait of a red fox in a forest" \
                          --neg "blurry, lowres, deformed" --seed 42 --steps 12 --guidance 7.5 \
                          --hires --hires-scale 1.5 --hires-strength 0.5 --cancel-after-steps 15
        [ -f "$WORK/restore.png" ] || { echo "  native follow-up not produced" >&2; exit 1; }
        verify_one c1.png "$WORK/restore.png" && echo "cancel-restore OK ✅" \
            || { echo "cancel-restore FAILED ❌ — ctx dims not restored on cancel" >&2; exit 1; }
        ;;
    *)
        echo "usage: $0 {capture|verify|determinism|hires-repro|cancel-restore}" >&2
        echo "       [--model DIR] [--bin PATH] [--mode exact|tolerance] [--tol N] [--max-frac F]" >&2
        exit 2 ;;
esac
