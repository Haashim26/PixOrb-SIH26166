# PixOrb implementation ↔ submitted SIH idea

| Submitted approach | Final implementation |
|---|---|
| Chandrayaan-2 OHRC / TMC-2 / IIRS + LRO NAC / SELENE inputs | `pixorb/ingestion.py` supports common raster images and GeoTIFF/COG through rasterio, with metadata extraction and illumination normalization |
| Coarse alignment: multi-resolution pyramid + phase correlation | `pixorb/coarse_align.py` builds a pyramid, evaluates scale/rotation hypotheses on usable levels, then performs phase-correlation translation refinement |
| Learned feature matching: SuperPoint + LightGlue / LoFTR | `pixorb/learned_match.py` uses SuperPoint + LightGlue first, LoFTR as learned fallback |
| Geometric verification: affine / homography + RANSAC | `pixorb/geometry.py` + `pixorb/pipeline.py` |
| Fine refinement: NCC + sub-pixel interpolation + grid-based ANMS | `pixorb/refine.py` |
| Outputs: registered image, matched keypoints CSV/JSON, 3x3 transform, RMSE/inlier ratio | `pixorb/evaluate.py` + desktop/web UI |
| Evaluation harness | `scripts/regression_sample.py`, `tests/`, `results/sample_run/` |

The original submitted PPT is included as `docs/SIH2026-IDEA-1.pptx`.

## Scope discipline

The implementation is designed for the multi-source lunar registration problem, but numerical results from the included sample are regression evidence only. Universal performance across every mission pair requires a held-out benchmark supplied by the organizers.
