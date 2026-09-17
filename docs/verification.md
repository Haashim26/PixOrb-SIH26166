# Verification status

The final code has been structurally verified against the SIH idea and the supplied team artifacts.

## Required acceptance tests

- [x] GeoTIFF/COG + common raster ingestion path
- [x] Metadata extraction when available
- [x] Multi-resolution pyramid
- [x] Phase-correlation coarse alignment
- [x] SuperPoint + LightGlue primary matcher interface
- [x] Classical fallback when learned backend is unavailable/low confidence
- [x] RANSAC affine/homography verification
- [x] NCC sub-pixel refinement with parabolic peak interpolation
- [x] Grid-based ANMS spatial distribution
- [x] Registered image output
- [x] Match CSV/JSON output
- [x] 3x3 transform output
- [x] RMSE/inlier-count/inlier-ratio output

## Regression fixture

The original team handoff for one 800x800 pair reports 34 candidate matches, 31 final matches and approximately 0.12 px mean reprojection error for its supplied Part-4 homography. This remains a historical regression fixture; the final product does not hard-code those values.

## Important limitation

A full scientific claim across every OHRC/TMC-2/IIRS/LRO/SELENE pair requires the released organiser dataset and a held-out benchmark. The software is designed for those inputs, but no responsible implementation can claim universal sub-pixel performance from one sample pair.
