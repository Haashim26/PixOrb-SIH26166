"""SIH26166 Stage 5: NCC sub-pixel refinement + grid-based ANMS."""
import cv2
import numpy as np
from .geometry import project


def _ncc(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    a = a - a.mean(); b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.sum(a * b) / den) if den > 1e-8 else -1.0


def _patch(img, x, y, r):
    xi, yi = int(round(float(x))), int(round(float(y)))
    if xi-r < 0 or yi-r < 0 or xi+r >= img.shape[1] or yi+r >= img.shape[0]:
        return None
    return img[yi-r:yi+r+1, xi-r:xi+r+1]


def subpixel_refine(reference, source, H, matches, patch=7, search=3):
    """Refine the reference-side coordinate with NCC and 1-D parabolic interpolation.

    H maps the supplied source image coordinates to reference coordinates. The source
    coordinate is retained because the local template is sampled there; the predicted
    reference location is optimized to sub-pixel precision.
    """
    refined = []
    for m in matches:
        sx, sy, rx, ry, conf = map(float, m[:5])
        pred = project(H, [[sx, sy]])[0]
        src_patch = _patch(source, sx, sy, patch)
        if src_patch is None:
            refined.append((sx, sy, rx, ry, conf)); continue

        best = -2.0; bx = by = 0
        for dy in range(-search, search + 1):
            for dx in range(-search, search + 1):
                rp = _patch(reference, pred[0] + dx, pred[1] + dy, patch)
                if rp is None: continue
                s = _ncc(src_patch, rp)
                if s > best: best, bx, by = s, dx, dy

        def parabola(y1, y2, y3):
            den = y1 - 2.0*y2 + y3
            if abs(den) < 1e-9: return 0.0
            return float(np.clip(0.5*(y1-y3)/den, -0.5, 0.5))

        def score(dx, dy):
            rp = _patch(reference, pred[0] + dx, pred[1] + dy, patch)
            return _ncc(src_patch, rp) if rp is not None else -1.0

        ox = oy = 0.0
        if -search < bx < search:
            ox = parabola(score(bx-1, by), score(bx, by), score(bx+1, by))
        if -search < by < search:
            oy = parabola(score(bx, by-1), score(bx, by), score(bx, by+1))

        nrx = float(pred[0] + bx + ox)
        nry = float(pred[1] + by + oy)
        refined.append((sx, sy, nrx, nry, float(max(conf, best))))
    return refined


def _anms_radius(points_xy, scores):
    """Compute the classic ANMS suppression radius for each point.

    For every point, radius is the distance to the nearest point with strictly
    higher confidence. Higher radius therefore means the point is both strong and
    spatially non-redundant.
    """
    p = np.asarray(points_xy, dtype=np.float32)
    s = np.asarray(scores, dtype=np.float32)
    n = len(p)
    if n == 0: return np.empty(0, np.float32)
    order = np.argsort(-s, kind="mergesort")
    radii = np.full(n, np.inf, dtype=np.float32)
    for rank in range(1, n):
        i = order[rank]
        stronger = p[order[:rank]]
        d2 = np.sum((stronger - p[i])**2, axis=1)
        if len(d2): radii[i] = np.sqrt(float(np.min(d2)))
    # The strongest point gets an infinite radius; keep it deterministic.
    if n > 1:
        strongest = order[0]
        d2 = np.sum((p[order[1:]] - p[strongest])**2, axis=1)
        radii[strongest] = np.sqrt(float(np.max(d2))) if len(d2) else np.inf
    return radii


def grid_anms(matches, shape, grid=(6, 6), per_cell=8, min_distance=0.0):
    """Grid-constrained ANMS.

    ANMS suppression radii are computed globally, then candidates are distributed
    across a reference-image grid. Within each cell, points with the largest ANMS
    radii are preferred. This directly targets the SIH requirement for spatially
    distributed correspondences instead of selecting only the highest-confidence
    clustered crater-rim features.
    """
    if not matches: return []
    h, w = shape[:2]; rows, cols = grid
    pts = np.asarray([[m[2], m[3]] for m in matches], np.float32)
    scores = np.asarray([m[4] for m in matches], np.float32)
    radii = _anms_radius(pts, scores)

    buckets = {(r, c): [] for r in range(rows) for c in range(cols)}
    for i, m in enumerate(matches):
        c = min(cols-1, max(0, int(float(m[2]) / max(w,1) * cols)))
        r = min(rows-1, max(0, int(float(m[3]) / max(h,1) * rows)))
        buckets[(r,c)].append(i)

    kept = []
    for key, idxs in buckets.items():
        idxs.sort(key=lambda i: (float(radii[i]), float(scores[i])), reverse=True)
        selected = []
        for i in idxs:
            if len(selected) >= per_cell: break
            if min_distance > 0 and any(np.linalg.norm(pts[i]-pts[j]) < min_distance for j in selected):
                continue
            selected.append(i)
        kept.extend(selected)
    kept.sort(key=lambda i: (int(min(rows-1, max(0, int(float(matches[i][3]) / max(h,1) * rows)))),
                           int(min(cols-1, max(0, int(float(matches[i][2]) / max(w,1) * cols)))),
                           -float(radii[i])))
    return [matches[i] for i in kept]
