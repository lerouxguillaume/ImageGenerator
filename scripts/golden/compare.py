#!/usr/bin/env python3
"""Per-pixel image compare for the golden harness's tolerance mode.

Exact (byte) comparison is done directly in the shell driver with `cmp`; this
script is only invoked when a machine's EP is not bit-deterministic and a small
documented per-pixel tolerance is needed instead. Exit 0 = within tolerance.
"""
import argparse
import sys


def load(path):
    try:
        import cv2
        import numpy as np
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"cv2 could not read {path}")
        return np.asarray(img, dtype=np.int32)
    except ImportError:
        pass
    try:
        import numpy as np
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.int32)
    except ImportError:
        print("compare.py: tolerance mode needs cv2 or Pillow (+numpy)", file=sys.stderr)
        sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tolerance", type=int, default=0,
                    help="max allowed per-channel absolute pixel difference")
    ap.add_argument("--max-frac", type=float, default=0.0,
                    help="max fraction of pixels allowed to exceed --tolerance")
    args = ap.parse_args()

    import numpy as np
    A, B = load(args.a), load(args.b)
    if A.shape != B.shape:
        print(f"FAIL(tolerance) shape {A.shape} != {B.shape}")
        sys.exit(1)

    diff = np.abs(A - B)
    max_abs = int(diff.max())
    frac_over = float((diff > args.tolerance).mean())
    ok = max_abs <= args.tolerance or frac_over <= args.max_frac
    verdict = "PASS(tolerance)" if ok else "FAIL(tolerance)"
    print(f"{verdict} max_abs_diff={max_abs} frac_over_tol={frac_over:.6f} "
          f"(tol={args.tolerance}, max_frac={args.max_frac})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
