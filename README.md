# PixOrb — SIH26166 Final Product

**Smart India Hackathon 2026 · ISRO / Space Technology**

**Problem Statement:** SIH26166 — Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS).

PixOrb is an end-to-end lunar image registration product based on the team's submitted technical approach. It supports multi-source lunar imagery, coarse scale/position alignment, learned correspondence, robust geometric verification, sub-pixel refinement, spatially distributed final correspondences, registration, and quantitative evaluation.

## Final pipeline

**Input**
→ multi-format ingestion + metadata + illumination normalization
→ **multi-resolution pyramid + phase correlation** for coarse scale/shift estimation
→ **SuperPoint + LightGlue** primary learned matching
→ **LoFTR** learned fallback, then SIFT only as a last-resort compatibility fallback
→ **affine/homography + RANSAC** geometric verification
→ **NCC + parabolic sub-pixel interpolation**
→ **grid-based ANMS** spatial distribution
→ final registration + RMSE/inlier/coverage evaluation
→ registered image, correspondences, transform and metrics.

This follows the technology and processing sequence shown in the submitted PixOrb technical approach: Python 3.11, OpenCV, PyTorch, SuperPoint + LightGlue, GDAL/rasterio, NumPy/SciPy, RANSAC and Matplotlib; the slide specifies multi-resolution coarse alignment, learned matching, affine/homography + RANSAC, NCC + sub-pixel interpolation + grid-based ANMS, and the corresponding evaluation outputs.

## Applications

### Desktop final application

```powershell
python app.py
```

or on Windows:

```powershell
.\run_pixorb.bat
```

The desktop UI is PySide6-based and works offline after model weights are cached.

### Shareable web application

The `web/` application uses the **same core pipeline**, with no duplicate registration implementation.

```powershell
python -m uvicorn web.main:app --host 127.0.0.1 --port 7860
```

See `WEB_DEPLOYMENT.md` for public hosting instructions.

## Windows setup

```powershell
.\setup_windows.ps1
```

The setup script creates Python 3.11 environment, installs the scientific/GUI stack, CPU PyTorch and the LightGlue package.

## Validation

The supplied team sample pair is included in `data/sample_pair/`. It is a regression fixture, not a hard-coded answer. The original Part-4 supplied fit measured about 0.12 px mean reprojection error for that specific pair. The current learned end-to-end regression is independently recomputed and reports its own metrics.

A single pair must not be presented as proof of universal performance across all OHRC/TMC-2/IIRS/LRO/SELENE imagery; that requires a held-out benchmark dataset.
