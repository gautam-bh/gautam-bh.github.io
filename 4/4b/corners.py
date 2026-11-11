import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from harris import get_harris_corners


def to_gray(im):
    im = np.asarray(im)
    if im.ndim == 3:
        im = rgb2gray(im[..., :3])
    elif im.dtype == np.uint8:
        im = im.astype(float) / 255.0
    return im.astype(float)


def detect_corners(im, edge_discard=20):
    return get_harris_corners(to_gray(im), edge_discard=edge_discard)


def plot_corners(im, coords, title=None, ax=None, out_path=None, s=4, color="r"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(im, cmap=None if np.asarray(im).ndim == 3 else "gray")
    ax.scatter(coords[1], coords[0], s=s, c=color, marker="o", linewidths=0)
    ax.set_title(title or f"{coords.shape[1]} corners")
    ax.axis("off")
    if out_path is not None:
        ax.figure.savefig(out_path, bbox_inches="tight", dpi=150)
    return ax


def strongest(h, coords, n):
    """Keep the n corners with the highest Harris response."""
    strength = h[coords[0], coords[1]]
    order = np.argsort(-strength)[:n]
    return coords[:, order]


def anms(h, coords, n_points=500, c_robust=0.9, max_candidates=5000, chunk=1000):
    coords = strongest(h, coords, max_candidates)
    f = h[coords[0], coords[1]]
    pts = coords.T.astype(np.float32)
    n = len(pts)

    radii = np.full(n, np.inf, dtype=np.float32)
    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        d2 = ((pts[sl, None, :] - pts[None, :, :]) ** 2).sum(-1)
        stronger = f[sl, None] < c_robust * f[None, :]
        d2[~stronger] = np.inf
        radii[sl] = np.sqrt(d2.min(axis=1))

    keep = np.argsort(-radii, kind="stable")[:n_points]
    return coords[:, keep], radii[keep]
