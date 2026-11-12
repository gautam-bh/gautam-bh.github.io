import numpy as np
from homography import computeH, apply_H


def ransac_homography(p1, p2, n_iters=5000, eps=4.0, seed=0):
    """4-point RANSAC for a robust homography mapping p1 -> p2.

    p1, p2: m x 2 arrays of matched (x, y) points (row k of p1 matches row k of p2).

    Repeat n_iters times:
      1. Pick 4 random matches.
      2. Compute the exact homography H through those 4 pairs.
      3. Count the inliers: matches with ||H p1 - p2|| < eps pixels.
    Keep the largest inlier set, then refit H by least squares on all of its
    inliers.

    Returns (H, inliers): H is 3x3, inliers is a boolean mask over the matches.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    m = len(p1)
    if m < 4:
        raise ValueError(f"need at least 4 matches for RANSAC, got {m}")

    rng = np.random.default_rng(seed)
    best = np.zeros(m, dtype=bool)
    for _ in range(n_iters):
        sample = rng.choice(m, 4, replace=False)
        try:
            H = computeH(p1[sample], p2[sample])
        except np.linalg.LinAlgError:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            err = np.linalg.norm(apply_H(H, p1) - p2, axis=1)
        inliers = err < eps          # NaN (degenerate H) compares False
        if inliers.sum() > best.sum():
            best = inliers

    if best.sum() < 4:
        raise RuntimeError("RANSAC found fewer than 4 inliers")
    H = computeH(p1[best], p2[best])
    return H, best
