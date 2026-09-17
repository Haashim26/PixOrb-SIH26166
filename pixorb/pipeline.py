"""PixOrb SIH26166 end-to-end registration pipeline."""
from pathlib import Path
import cv2, numpy as np
from .ingestion import load_image
from .coarse_align import coarse_align
from .learned_match import match_images
from .geometry import ransac_verify, residuals, project
from .refine import subpixel_refine, grid_anms
from .evaluate import evaluate, save_matches, save_metrics


def _to_original_matches(matches, Hc):
    """Convert source coordinates from coarse-aligned canvas back to original source pixels."""
    if not matches: return []
    Hci = np.linalg.inv(Hc)
    src = project(Hci, [[m[0],m[1]] for m in matches])
    out=[]
    for m,p in zip(matches,src):
        out.append((float(p[0]),float(p[1]),float(m[2]),float(m[3]),float(m[4])))
    return out


def run(reference_path, source_path, output_dir, model="homography", threshold=3.0):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    ref=load_image(reference_path); src=load_image(source_path)
    aligned,Hc,coarse_info=coarse_align(ref.image,src.image)
    raw,match_info=match_images(ref.image,aligned)
    if len(raw)<4: raise RuntimeError(f"Only {len(raw)} candidate matches were found")

    Hfine, initial_inliers, mask=ransac_verify(raw,model=model,threshold=threshold)
    initial_err=residuals(Hfine,initial_inliers)

    refined=subpixel_refine(ref.image,aligned,Hfine,initial_inliers)
    final_aligned=grid_anms(refined,ref.image.shape,grid=(6,6),per_cell=10)
    if len(final_aligned)<4:
        final_aligned=refined
    if len(final_aligned)<4:
        raise RuntimeError("Too few correspondences remain after refinement/distribution")

    # Final geometric verification is performed in the same coordinate system as
    # the learned matcher: coarse-aligned source -> reference.
    Hfinal, final_inliers, final_mask=ransac_verify(final_aligned,model=model,threshold=threshold)
    if len(final_inliers)<4:
        Hfinal=Hfine; final_inliers=final_aligned

    # Compose the fine transform with the actual coarse source->reference transform.
    Htotal=Hfinal @ Hc
    Htotal=Htotal/Htotal[2,2]
    final_original=_to_original_matches(final_inliers,Hc)

    warped,metrics,overlap,overlay,diff8=evaluate(Htotal,final_original,len(raw),ref.image,src.image)
    metrics.update({
        "backend":match_info.get("backend"),
        "learned_available":match_info.get("learned_available"),
        "fallback_used":match_info.get("backend") != "SuperPoint+LightGlue",
        "learned_backend_fallback":match_info.get("learned_backend_fallback", False),
        "fallback_reason":match_info.get("fallback_reason", ""),
        "deduplicated_candidates":len(raw),
        "initial_ransac_inlier_count":len(initial_inliers),
        "initial_ransac_inlier_ratio":len(initial_inliers)/len(raw),
        "ransac_inlier_count":len(initial_inliers),
        "ransac_inlier_ratio":len(initial_inliers)/len(raw),
        "final_ransac_inlier_count":len(final_inliers),
        "final_ransac_inlier_ratio":len(final_inliers)/len(final_aligned),
        "overall_retained_ratio":len(final_inliers)/len(raw),
        "subpixel_changed_count":sum(1 for a,b in zip(initial_inliers,refined) if abs(a[2]-b[2])>1e-6 or abs(a[3]-b[3])>1e-6),
        "model":model,"ransac_threshold_px":threshold,
        "coarse_alignment":coarse_info,
        "reference_metadata":ref.metadata,
        "source_metadata":src.metadata,
    })
    cv2.imwrite(str(out/"registered.png"),warped)
    cv2.imwrite(str(out/"reference.png"),ref.image)
    cv2.imwrite(str(out/"source_original.png"),src.image)
    cv2.imwrite(str(out/"source_coarse_aligned.png"),aligned)
    cv2.imwrite(str(out/"overlap_mask.png"),overlap)
    cv2.imwrite(str(out/"overlay.png"),overlay)
    cv2.imwrite(str(out/"difference_map.png"),diff8)
    np.save(str(out/"transform_coarse_source_to_reference.npy"),Hc)
    np.save(str(out/"transform_fine_aligned_to_reference.npy"),Hfinal)
    np.save(str(out/"transform_source_to_reference.npy"),Htotal)
    save_matches(out/"match_points.json",final_original); save_metrics(out/"metrics.json",metrics)
    (out/"transform_source_to_reference.txt").write_text(np.array2string(Htotal,precision=10))
    # Correspondence visualization:
    # Keep every verified match in JSON/CSV, but display a representative
    # spatially distributed subset so the figure remains readable.
    #
    # The visualization copy of the coarse-aligned source is cropped and
    # resized only for presentation. The actual registration coordinates,
    # transforms and saved match data are unchanged.

    rows, cols = 6, 6
    per_cell_display = 1
    h, w = ref.image.shape[:2]

    cells = {}

    for m in final_inliers:
        x_ref, y_ref = float(m[2]), float(m[3])

        c = min(
            cols - 1,
            max(0, int(x_ref / max(w, 1) * cols))
        )
        r = min(
            rows - 1,
            max(0, int(y_ref / max(h, 1) * rows))
        )

        cells.setdefault((r, c), []).append(m)

    display_matches = []

    for cell_matches in cells.values():
        # Highest-confidence match is preferred for visualization.
        cell_matches = sorted(
            cell_matches,
            key=lambda m: float(m[4]),
            reverse=True
        )
        display_matches.extend(cell_matches[:per_cell_display])

    # Crop black/empty margins from the visualization copy.
    src_valid = (aligned > 5).astype(np.uint8) * 255
    ys, xs = np.where(src_valid > 0)

    if len(xs) > 0 and len(ys) > 0:
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1

        aligned_vis = aligned[y0:y1, x0:x1]

        # Fit the cropped source into the same display height as reference.
        display_h = h
        display_w = max(
            1,
            int(round(aligned_vis.shape[1] * display_h / aligned_vis.shape[0]))
        )

        aligned_vis = cv2.resize(
            aligned_vis,
            (display_w, display_h),
            interpolation=cv2.INTER_LINEAR
        )

        scale_x = display_w / max(aligned_vis.shape[1], 1)
        scale_y = display_h / max(aligned_vis.shape[0], 1)

        # Since the resize above already produced display_h x display_w,
        # calculate the coordinate scale from the original crop dimensions.
        crop_w = max(x1 - x0, 1)
        crop_h = max(y1 - y0, 1)

        scale_x = display_w / crop_w
        scale_y = display_h / crop_h

    else:
        x0, y0 = 0, 0
        aligned_vis = aligned.copy()

        if aligned_vis.shape[:2] != (h, w):
            aligned_vis = cv2.resize(
                aligned_vis,
                (w, h),
                interpolation=cv2.INTER_LINEAR
            )

        display_w = aligned_vis.shape[1]
        crop_w = aligned.shape[1]
        crop_h = aligned.shape[0]

        scale_x = display_w / max(crop_w, 1)
        scale_y = h / max(crop_h, 1)

    vis = np.hstack([ref.image, aligned_vis])
    off = ref.image.shape[1]

    for m in display_matches:
        x_src, y_src, x_ref, y_ref = map(float, m[:4])

        p_ref = (
            int(round(x_ref)),
            int(round(y_ref))
        )

        p_src = (
            int(round((x_src - x0) * scale_x)) + off,
            int(round((y_src - y0) * scale_y))
        )

        cv2.line(
            vis,
            p_ref,
            p_src,
            255,
            1
        )

        cv2.circle(
            vis,
            p_ref,
            3,
            255,
            -1
        )

        cv2.circle(
            vis,
            p_src,
            3,
            255,
            -1
        )

    cv2.imwrite(str(out / "matches.png"), vis)
        
    return {"reference":ref,"source":src,"coarse_aligned":aligned,"transform":Htotal,"matches":final_original,"metrics":metrics,"registered":warped,"overlay":overlay,"difference_map":diff8,"output_dir":str(out)}
