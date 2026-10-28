import json
import numpy as np
import matplotlib.pyplot as plt
from homography import computeH, apply_H

MAX_OUTPUT_PIXELS = 40_000_000


def to_float(im):
    im = np.asarray(im)
    if im.dtype == np.uint8:
        return im.astype(float) / 255.0
    return im.astype(float)


def warped_bbox(im, H):
    h, w = im.shape[:2]
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=float)
    c = apply_H(H, corners)
    xmin, ymin = np.floor(c.min(axis=0)).astype(int).tolist()
    xmax, ymax = np.ceil(c.max(axis=0)).astype(int).tolist()
    return xmin, ymin, xmax, ymax


def source_coords(im, H, bbox=None):
    xmin, ymin, xmax, ymax = warped_bbox(im, H) if bbox is None else bbox
    n_pixels = (xmax - xmin + 1) * (ymax - ymin + 1)
    if n_pixels > MAX_OUTPUT_PIXELS:
        raise ValueError(f"warped output would be {n_pixels:,} pixels; pass a smaller bbox")
    xs, ys = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        src = apply_H(np.linalg.inv(H), np.stack([xs.ravel(), ys.ravel()], axis=1))
    src[~np.isfinite(src)] = -1
    return src[:, 0].reshape(xs.shape), src[:, 1].reshape(xs.shape), (xmin, ymin)


def warpImageNearestNeighbor(im, H, bbox=None):
    im = to_float(im)
    gray = im.ndim == 2
    if gray:
        im = im[..., None]
    h, w = im.shape[:2]
    sx, sy, offset = source_coords(im, H, bbox)

    xi = np.round(sx).astype(int)
    yi = np.round(sy).astype(int)
    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)

    out = np.zeros(sx.shape + (im.shape[2],))
    out[valid] = im[yi[valid], xi[valid]]
    if gray:
        out = out[..., 0]
    return out, valid.astype(float), offset


def warpImageBilinear(im, H, bbox=None):
    im = to_float(im)
    gray = im.ndim == 2
    if gray:
        im = im[..., None]
    h, w = im.shape[:2]
    sx, sy, offset = source_coords(im, H, bbox)

    valid = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    x, y = sx[valid], sy[valid]
    x0 = np.clip(np.floor(x).astype(int), 0, max(w - 2, 0))
    y0 = np.clip(np.floor(y).astype(int), 0, max(h - 2, 0))
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    dx = (x - x0)[:, None]
    dy = (y - y0)[:, None]

    out = np.zeros(sx.shape + (im.shape[2],))
    out[valid] = ((1 - dx) * (1 - dy) * im[y0, x0] + dx * (1 - dy) * im[y0, x1]
                  + (1 - dx) * dy * im[y1, x0] + dx * dy * im[y1, x1])
    if gray:
        out = out[..., 0]
    return out, valid.astype(float), offset


def rectangle_size(corners):
    tl, tr, br, bl = np.asarray(corners, dtype=float)
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    return int(round(width)), int(round(height))


def rectify(im, corners, width=None, height=None, warp=warpImageBilinear, margin=None):
    if width is None or height is None:
        width, height = rectangle_size(corners)
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=float)
    H = computeH(corners, target)
    bbox = None if margin is None else (-margin, -margin, width - 1 + margin, height - 1 + margin)
    out, alpha, offset = warp(im, H, bbox)
    return out, alpha, H


def select_corners(im):
    plt.figure(figsize=(10, 8))
    plt.imshow(im)
    plt.title("Click 4 corners: top-left, top-right, bottom-right, bottom-left")
    pts = np.array(plt.ginput(4, timeout=0))
    plt.close()
    return pts


def save_corners(path, corners):
    with open(path, "w") as f:
        json.dump({"corners": np.asarray(corners).tolist()}, f, indent=1)


def load_corners(path):
    with open(path) as f:
        return np.array(json.load(f)["corners"], dtype=float)


def with_alpha(im, alpha):
    if im.ndim == 2:
        im = np.stack([im] * 3, axis=-1)
    return np.dstack([np.clip(im, 0, 1), alpha])
