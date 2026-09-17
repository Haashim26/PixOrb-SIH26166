# PixOrb Architecture — SIH26166

The implementation follows the submitted PixOrb idea exactly at the algorithmic level:

**Input (Chandrayaan-2 OHRC/TMC-2/IIRS + LRO NAC/SELENE reference)**
→ **multi-format ingestion + metadata + illumination normalization**
→ **multi-resolution pyramid + phase-correlation coarse alignment**
→ **SuperPoint + LightGlue learned correspondence**
→ **affine/homography RANSAC verification**
→ **NCC + parabolic sub-pixel refinement**
→ **grid-based ANMS spatial distribution**
→ **warp + RMSE/inlier-count/inlier-ratio evaluation**
→ **registered product + match points + transform + metrics**.

The architecture intentionally includes the hybrid fallback described in the feasibility slide: when the learned matcher cannot initialize or does not provide enough reliable correspondences, a classical descriptor matcher is used rather than producing no result. The UI clearly records which backend produced the result.

## Data contract

- Ingestion: `ImageRecord(image, metadata, path)`
- Coarse alignment: `aligned_image, 3x3 transform, info`
- Matching: list of `(x_source, y_source, x_reference, y_reference, confidence)`
- Verification: RANSAC inliers + 3x3 transform
- Refinement: floating-point/sub-pixel matches + final transform
- Output: registered image, JSON/CSV match points, transform matrix, metrics JSON

## Why the old SIFT result is not the final algorithm

The supplied student work used SIFT+FLANN and simple cornerSubPix. Those artifacts are valuable for regression/debugging, but the SIH idea explicitly specifies learned SuperPoint+LightGlue matching and NCC/parabolic sub-pixel refinement with ANMS. The final PixOrb implementation therefore uses those methods and keeps SIFT only as the documented hybrid fallback.
