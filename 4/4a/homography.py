import json
import numpy as np
import matplotlib.pyplot as plt


def build_system(im1_pts, im2_pts):
    im1_pts = np.asarray(im1_pts, dtype=float)
    im2_pts = np.asarray(im2_pts, dtype=float)
    n = im1_pts.shape[0]
    A = np.zeros((2 * n, 8))
    b = np.zeros(2 * n)
    for i, ((x, y), (xp, yp)) in enumerate(zip(im1_pts, im2_pts)):
        A[2 * i] = [x, y, 1, 0, 0, 0, -x * xp, -y * xp]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -x * yp, -y * yp]
        b[2 * i] = xp
        b[2 * i + 1] = yp
    return A, b


def computeH(im1_pts, im2_pts):
    if len(im1_pts) != len(im2_pts):
        raise ValueError("im1_pts and im2_pts must have the same length")
    if len(im1_pts) < 4:
        raise ValueError("need at least 4 correspondences")
    A, b = build_system(im1_pts, im2_pts)
    h, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.append(h, 1).reshape(3, 3)


def apply_H(H, pts):
    pts = np.asarray(pts, dtype=float)
    homog = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return homog[:, :2] / homog[:, 2:]


def reprojection_error(H, im1_pts, im2_pts):
    return np.linalg.norm(apply_H(H, im1_pts) - np.asarray(im2_pts, dtype=float), axis=1)


def select_points(im1, im2, n):
    pts = []
    for k, im in enumerate([im1, im2], start=1):
        plt.figure(figsize=(10, 8))
        plt.imshow(im)
        plt.title(f"Image {k}: click {n} points (same order in both images)")
        pts.append(np.array(plt.ginput(n, timeout=0)))
        plt.close()
    return pts[0], pts[1]


def save_points(path, im1_pts, im2_pts):
    with open(path, "w") as f:
        json.dump({"im1Points": np.asarray(im1_pts).tolist(),
                   "im2Points": np.asarray(im2_pts).tolist()}, f, indent=1)


def load_points(path):
    with open(path) as f:
        d = json.load(f)
    return np.array(d["im1Points"], dtype=float), np.array(d["im2Points"], dtype=float)


def plot_correspondences(im1, im2, im1_pts, im2_pts, out_path=None):
    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    for a, im, pts in [(ax[0], im1, im1_pts), (ax[1], im2, im2_pts)]:
        a.imshow(im)
        a.scatter(pts[:, 0], pts[:, 1], c="red", s=25)
        for i, (x, y) in enumerate(pts):
            a.annotate(str(i), (x, y), color="yellow", fontsize=11,
                       xytext=(4, -4), textcoords="offset points")
        a.axis("off")
    ax[0].set_title("Image 1")
    ax[1].set_title("Image 2")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.show()
