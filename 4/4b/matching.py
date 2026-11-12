import numpy as np
import matplotlib.pyplot as plt
from harris import dist2

RATIO_THRESHOLD = 0.5


def match_features(desc1, desc2, ratio=RATIO_THRESHOLD):
    """Feature matching with Lowe's ratio test (MOPS Section 5.1).

    The matching error between two bias/gain-normalized descriptors is the
    sum of squared differences, e = sum_x n(x)^2 = ||d1 - d2||^2.
    For each feature in image 1:
        e_1NN = error to its nearest neighbour in image 2
        e_2NN = error to its second nearest neighbour in image 2
    The 2-NN is assumed to be an incorrect match, so e_2NN estimates the
    "outlier distance". The match is kept if e_1NN / e_2NN < ratio.

    Returns (matches, ratios):
      matches: m x 2 array of index pairs (i into desc1, j into desc2)
      ratios:  e_1NN / e_2NN for every feature in desc1 (useful for plotting)
    """
    E = dist2(desc1, desc2)          
    nn = np.argsort(E, axis=1)[:, :2]
    rows = np.arange(len(desc1))
    e1 = E[rows, nn[:, 0]]
    e2 = E[rows, nn[:, 1]]
    ratios = e1 / np.maximum(e2, 1e-12)

    keep = ratios < ratio
    matches = np.stack([rows[keep], nn[keep, 0]], axis=1)
    return matches, ratios


def matched_points(coords1, coords2, matches):
    """Turn index pairs into (x, y) point arrays, the format computeH uses."""
    p1 = coords1[:, matches[:, 0]][::-1].T.astype(float)   # (ys, xs) -> (x, y)
    p2 = coords2[:, matches[:, 1]][::-1].T.astype(float)
    return p1, p2


def plot_matches(im1, im2, p1, p2, title=None, ax=None, out_path=None):
    """Show the two images side by side with a line for every match."""
    h = max(im1.shape[0], im2.shape[0])
    pad = lambda im: np.pad(im, ((0, h - im.shape[0]), (0, 0), (0, 0)))
    canvas = np.hstack([pad(im1), pad(im2)])
    off = im1.shape[1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(canvas)
    colors = plt.cm.hsv(np.linspace(0, 1, len(p1), endpoint=False))
    for (x1, y1), (x2, y2), c in zip(p1, p2, colors):
        ax.plot([x1, x2 + off], [y1, y2], "-", color=c, lw=0.8)
        ax.plot([x1, x2 + off], [y1, y2], "o", color=c, ms=3)
    ax.set_title(title or f"{len(p1)} matches")
    ax.axis("off")
    if out_path is not None:
        ax.figure.savefig(out_path, bbox_inches="tight", dpi=150)
    return ax
