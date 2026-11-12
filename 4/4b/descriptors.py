import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates
from corners import to_gray

WINDOW = 40   # size of the window around each corner, in pixels
PATCH = 8     # descriptor is PATCH x PATCH
SPACING = WINDOW // PATCH   # sample every 5 pixels
BLUR_SIGMA = 2.0            # anti-alias before taking every 5th pixel


def extract_descriptors(im, coords, window=WINDOW, patch=PATCH, sigma=BLUR_SIGMA, eps=1e-8):
    """Axis-aligned MOPS descriptors (Section 4 of Brown et al.), no rotation.

    For each corner (y, x) in coords (2 x n, as returned by harris/anms):
      1. Blur the grayscale image so that subsampling does not alias.
      2. Take the window x window region centered on the corner and sample it
         every window/patch pixels, giving a patch x patch descriptor.
      3. Bias/gain normalize: subtract the mean, divide by the std.

    Corners whose window would leave the image are dropped.

    Returns (desc, coords): desc is n x (patch*patch), coords is 2 x n for the
    corners that were kept (same order as desc).
    """
    gray = gaussian_filter(to_gray(im), sigma)
    H, W = gray.shape
    spacing = window // patch
    half = window // 2

    ys, xs = np.asarray(coords)
    ok = (ys - half >= 0) & (ys + half <= H) & (xs - half >= 0) & (xs + half <= W)
    ys, xs = ys[ok], xs[ok]

    # sample at the center of each spacing x spacing cell of the window,
    # i.e. offsets -17.5, -12.5, ..., 17.5 relative to the corner (rounded to pixels)
    offsets = np.arange(patch) * spacing - half + spacing // 2
    rows = ys[:, None, None] + offsets[None, :, None]
    cols = xs[:, None, None] + offsets[None, None, :]
    patches = gray[rows, cols]                      # n x patch x patch

    desc = patches.reshape(len(ys), -1)
    desc = desc - desc.mean(axis=1, keepdims=True)
    desc = desc / (desc.std(axis=1, keepdims=True) + eps)
    return desc, np.stack([ys, xs])


def _window(im, y, x, theta, window):
    """window x window crop of im centred on (y, x), rotated by theta (bilinear)."""
    o = np.arange(window) - (window - 1) / 2
    du, dv = np.meshgrid(o, o)
    c, s = np.cos(theta), np.sin(theta)
    sx, sy = x + c * du - s * dv, y + s * du + c * dv
    im = np.asarray(im, dtype=float) / (255.0 if np.asarray(im).dtype == np.uint8 else 1.0)
    chans = [map_coordinates(im[..., k], [sy, sx], order=1) for k in range(im.shape[2])]
    return np.clip(np.stack(chans, axis=-1), 0, 1)


def show_descriptors(im, desc, coords, idx, thetas=None, window=WINDOW, patch=PATCH, title=None, out_path=None):
    """Plot the chosen corners on the image, their 40x40 windows, and the
    normalized 8x8 descriptors. If thetas is given, the windows are drawn and
    cropped in each corner's rotated frame, with an arrow showing theta."""
    k = len(idx)
    fig = plt.figure(figsize=(2.2 * k, 9))
    gs = fig.add_gridspec(3, k, height_ratios=[3, 1, 1])

    ax_im = fig.add_subplot(gs[0, :])
    ax_im.imshow(im)
    half = window // 2
    for n, i in enumerate(idx):
        y, x = coords[:, i]
        th = 0.0 if thetas is None else thetas[i]
        corner = np.array([[-half, -half], [half, -half], [half, half], [-half, half], [-half, -half]])
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        box = corner @ R.T + [x, y]
        ax_im.plot(box[:, 0], box[:, 1], "-", color="yellow", lw=1.5)
        if thetas is not None:
            ax_im.plot([x, x + half * np.cos(th)], [y, y + half * np.sin(th)], "-", color="cyan", lw=1.5)
        ax_im.text(x + half + 4, y, str(n + 1), color="yellow", fontsize=11, weight="bold", va="center")
    ax_im.set_title(title or "Sample features")
    ax_im.axis("off")

    for n, i in enumerate(idx):
        y, x = coords[:, i]
        th = 0.0 if thetas is None else thetas[i]
        ax = fig.add_subplot(gs[1, n])
        ax.imshow(_window(im, y, x, th, window), interpolation="nearest")
        ax.set_title(f"{n + 1}: 40x40 window" + ("" if thetas is None else f" ({np.degrees(th):.0f}°)"), fontsize=9)
        ax.axis("off")

        ax = fig.add_subplot(gs[2, n])
        ax.imshow(desc[i].reshape(patch, patch), cmap="gray", interpolation="nearest")
        ax.set_title(f"{n + 1}: 8x8 descriptor", fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Rotation invariance (MOPS Section 2): orient each corner by the blurred
# local gradient, then sample the descriptor in that rotated frame.
# ---------------------------------------------------------------------------
ORIENT_SIGMA = 4.5   # sigma_o from the paper


def orientations(im, coords, sigma_o=ORIENT_SIGMA):
    """theta for each corner, from u = grad of the image blurred with sigma_o:
    [cos theta, sin theta] = u / |u|.  coords is 2 x n (ys, xs)."""
    gray = to_gray(im)
    gy = gaussian_filter(gray, sigma_o, order=(1, 0))   # d/dy
    gx = gaussian_filter(gray, sigma_o, order=(0, 1))   # d/dx
    ys, xs = np.asarray(coords)
    return np.arctan2(gy[ys, xs], gx[ys, xs])


def extract_oriented_descriptors(im, coords, window=WINDOW, patch=PATCH, sigma=BLUR_SIGMA,
                                 sigma_o=ORIENT_SIGMA, eps=1e-8):
    """Rotation-invariant MOPS descriptors.

    Same as extract_descriptors, except the 8x8 sampling grid is rotated by
    each corner's orientation theta, so the patch's x-axis always points along
    the dominant gradient. Samples are bilinearly interpolated.

    Returns (desc, coords, thetas) for the corners whose rotated window stays
    inside the image.
    """
    gray = gaussian_filter(to_gray(im), sigma)
    H, W = gray.shape
    spacing = window / patch
    ys, xs = np.asarray(coords)
    thetas = orientations(im, coords, sigma_o)

    # sample offsets in the corner's own frame: -17.5, -12.5, ..., 17.5
    o = (np.arange(patch) - (patch - 1) / 2) * spacing
    du, dv = np.meshgrid(o, o)                         # du along patch x, dv along patch y
    c, s = np.cos(thetas)[:, None, None], np.sin(thetas)[:, None, None]
    sx = xs[:, None, None] + c * du - s * dv           # rotate by theta into image coords
    sy = ys[:, None, None] + s * du + c * dv

    ok = (sx.min((1, 2)) >= 0) & (sx.max((1, 2)) <= W - 1) & \
         (sy.min((1, 2)) >= 0) & (sy.max((1, 2)) <= H - 1)
    sx, sy = sx[ok], sy[ok]

    patches = map_coordinates(gray, [sy.ravel(), sx.ravel()], order=1).reshape(sx.shape)
    desc = patches.reshape(len(sx), -1)
    desc = desc - desc.mean(axis=1, keepdims=True)
    desc = desc / (desc.std(axis=1, keepdims=True) + eps)
    return desc, np.stack([ys[ok], xs[ok]]), thetas[ok]
