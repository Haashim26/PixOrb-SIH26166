"""SIH26166 Stage 6: registration, metrics, overlap-aware spatial evaluation and exports."""
import csv, json
from pathlib import Path
import cv2
import numpy as np
from .geometry import residuals, project


def _spatial_metrics(matches, overlap_mask, grid=(6,6)):
    rows, cols = grid
    h, w = overlap_mask.shape[:2]
    counts = np.zeros((rows, cols), dtype=np.int32)
    eligible = np.zeros((rows, cols), dtype=bool)
    mask = overlap_mask > 0
    for r in range(rows):
        y0, y1 = int(r*h/rows), int((r+1)*h/rows)
        for c in range(cols):
            x0, x1 = int(c*w/cols), int((c+1)*w/cols)
            eligible[r,c] = bool(np.any(mask[y0:y1, x0:x1]))
    for m in matches:
        x,y = float(m[2]), float(m[3])
        c = min(cols-1, max(0, int(x/max(w,1)*cols)))
        r = min(rows-1, max(0, int(y/max(h,1)*rows)))
        if eligible[r,c]: counts[r,c] += 1

    eligible_counts = counts[eligible]
    occupied = int(np.count_nonzero(eligible_counts > 0))
    eligible_n = int(np.count_nonzero(eligible))
    occ_ratio = occupied/eligible_n if eligible_n else 0.0
    if eligible_counts.sum() > 0:
        p = eligible_counts[eligible_counts > 0].astype(float)
        p /= p.sum()
        entropy = float(-np.sum(p*np.log(p)) / np.log(len(p))) if len(p) > 1 else 1.0
        mean = float(np.mean(eligible_counts)); cv = float(np.std(eligible_counts)/mean) if mean > 1e-12 else 0.0
    else:
        entropy, cv = 0.0, float("inf")
    return {
        "grid_rows": rows, "grid_cols": cols,
        "occupied_cells": occupied, "eligible_overlap_cells": eligible_n,
        "total_cells": rows*cols,
        "grid_occupancy_ratio": occ_ratio,
        "spatial_entropy_normalized": entropy,
        "grid_count_cv": cv,
        "grid_counts": counts.tolist(),
    }


def evaluate(H_total, matches, candidate_count, reference, original_source, grid=(6,6)):
    """Evaluate final correspondences and warp the ORIGINAL source using source->reference H."""
    e = residuals(H_total, matches)
    warped = cv2.warpPerspective(original_source, H_total, (reference.shape[1], reference.shape[0]))
    src_mask = (original_source > 5).astype(np.uint8)*255
    overlap = cv2.warpPerspective(src_mask, H_total, (reference.shape[1], reference.shape[0]))
    valid = overlap > 0
    diff = np.abs(reference.astype(np.float32)-warped.astype(np.float32))
    # Presentation/diagnostic products are computed from the valid geometric
    # overlap. Outside the overlap the reference is preserved rather than
    # blending against the black warp canvas.
    blend = cv2.addWeighted(reference, 0.5, warped, 0.5, 0)
    overlay = reference.copy()
    overlay[valid] = blend[valid]
    diff8 = np.zeros_like(reference, dtype=np.uint8)
    if valid.any():
        dvalid = diff[valid]
        lo = float(np.percentile(dvalid, 2))
        hi = float(np.percentile(dvalid, 98))
        if hi <= lo:
            hi = lo + 1.0
        diff8 = np.clip((diff - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
        diff8[~valid] = 0
    # Make the overlap boundary explicit without altering the registered data.
    boundary = cv2.morphologyEx(overlap, cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8))
    overlay[boundary > 0] = 255
    metrics = {
        "candidate_matches": int(candidate_count),
        "final_match_count": int(len(matches)),
        "inlier_count": int(len(matches)),
        "inlier_ratio": float(len(matches)/candidate_count) if candidate_count else 0.0,
        "rmse_px": float(np.sqrt(np.mean(e**2))) if len(e) else float("inf"),
        "mean_reprojection_error_px": float(np.mean(e)) if len(e) else float("inf"),
        "median_reprojection_error_px": float(np.median(e)) if len(e) else float("inf"),
        "max_reprojection_error_px": float(np.max(e)) if len(e) else float("inf"),
        "overlap_percent": float(valid.mean()*100),
        "overlap_mae": float(diff[valid].mean()) if valid.any() else float("nan"),
        "overlap_rmse": float(np.sqrt(np.mean(diff[valid]**2))) if valid.any() else float("nan"),
    }
    metrics.update(_spatial_metrics(matches, overlap, grid))
    return warped, metrics, overlap, overlay, diff8


def save_matches(path, matches):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for m in matches:
        if hasattr(m,'as_dict'): rows.append(m.as_dict())
        else: rows.append({"x_src":float(m[0]),"y_src":float(m[1]),"x_ref":float(m[2]),"y_ref":float(m[3]),"confidence":float(m[4])})
    path.write_text(json.dumps(rows,indent=2))
    with path.with_suffix('.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys() if rows else ['x_src','y_src','x_ref','y_ref','confidence']); w.writeheader(); w.writerows(rows)


def save_metrics(path,metrics):
    Path(path).write_text(json.dumps(metrics,indent=2,default=str))
