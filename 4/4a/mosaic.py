import numpy as np
from scipy.ndimage import gaussian_filter
from warp import to_float, warped_bbox, warpImageBilinear


def edge_weight(im):
    h, w = im.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    d = np.minimum.reduce([xs, w - 1 - xs, ys, h - 1 - ys]).astype(float) + 1
    return d / d.max()


def union_bbox(images, Hs):
    boxes = np.array([warped_bbox(im, H) for im, H in zip(images, Hs)])
    return (int(boxes[:, 0].min()), int(boxes[:, 1].min()),
            int(boxes[:, 2].max()), int(boxes[:, 3].max()))


def warp_all(images, Hs, warp=warpImageBilinear):
    bbox = union_bbox(images, Hs)
    layers = []
    for im, H in zip(images, Hs):
        out, alpha, _ = warp(to_float(im), H, bbox)
        weight, _, _ = warp(edge_weight(im), H, bbox)
        layers.append((out, alpha, weight * alpha))
    return layers, bbox


def blur_masked(im, alpha, sigma):
    blurred = gaussian_filter(im * alpha[..., None], sigma=(sigma, sigma, 0))
    norm = gaussian_filter(alpha, sigma=sigma)
    return blurred / np.maximum(norm, 1e-12)[..., None]


def blend_two_band(layers, sigma=8):
    lows = [blur_masked(out, alpha, sigma) for out, alpha, _ in layers]
    highs = [out - low for (out, _, _), low in zip(layers, lows)]
    weights = np.stack([w for _, _, w in layers])

    den = weights.sum(axis=0)
    low = sum(w[..., None] * l for w, l in zip(weights, lows)) / np.maximum(den, 1e-12)[..., None]

    best = weights.argmax(axis=0)
    high = np.zeros_like(low)
    for i, h in enumerate(highs):
        high[best == i] = h[best == i]

    return np.clip(low + high, 0, 1), (den > 0).astype(float)


def mosaic(images, Hs, warp=warpImageBilinear):
    layers, bbox = warp_all(images, Hs, warp)
    out, alpha = blend_two_band(layers)
    out[alpha == 0] = 0
    return out, alpha, layers, bbox
