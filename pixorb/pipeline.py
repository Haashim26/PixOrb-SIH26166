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

    vis=np.hstack([ref.image,aligned]); off=ref.image.shape[1]
    for m in final_inliers:
        cv2.line(vis,(int(round(m[2])),int(round(m[3]))),(int(round(m[0]))+off,int(round(m[1]))),255,1)
        cv2.circle(vis,(int(round(m[2])),int(round(m[3]))),3,255,-1)
        cv2.circle(vis,(int(round(m[0]))+off,int(round(m[1]))),3,255,-1)
    cv2.imwrite(str(out/"matches.png"),vis)
    return {"reference":ref,"source":src,"coarse_aligned":aligned,"transform":Htotal,"matches":final_original,"metrics":metrics,"registered":warped,"overlay":overlay,"difference_map":diff8,"output_dir":str(out)}
