"""SIH26166 Stage 1: multi-format ingestion and illumination normalization."""
from pathlib import Path
import json
import re
import numpy as np
import cv2
from .types import ImageRecord

try:
    import rasterio
except Exception:
    rasterio = None

def _norm01(a):
    a = np.asarray(a, np.float32)
    finite = np.isfinite(a)
    if not finite.any(): return np.zeros(a.shape, np.float32)
    lo, hi = np.percentile(a[finite], (1, 99))
    if hi <= lo: return np.zeros(a.shape, np.float32)
    return np.clip((a-lo)/(hi-lo), 0, 1)

def _illumination_normalize(gray):
    # Local background division reduces large-scale illumination/shadow differences.
    x = _norm01(gray)
    sigma = max(3.0, min(gray.shape)*0.03)
    bg = cv2.GaussianBlur(x, (0,0), sigmaX=sigma, sigmaY=sigma)
    y = x / (bg + 0.15)
    y = _norm01(y)
    return (255*y).astype(np.uint8)

def _metadata_from_name(path):
    name = Path(path).name.upper()
    sensor = "UNKNOWN"
    for s in ("OHRC", "TMC-2", "TMC", "IIRS", "LRO NAC", "NAC", "SELENE", "KAGUYA"):
        if s in name:
            sensor = s
            break
    return {"sensor_name": sensor}

def load_image(path, bands=None, illumination_normalize=True):
    """Load GeoTIFF/COG through rasterio; common raster formats through OpenCV.
    Returns an ImageRecord with normalized uint8 grayscale image and metadata.
    """
    path = str(path)
    suffix = Path(path).suffix.lower()
    meta = _metadata_from_name(path)
    if suffix in {".tif", ".tiff", ".cog"} and rasterio is not None:
        with rasterio.open(path) as ds:
            count = ds.count
            if bands is None:
                bands = list(range(1, min(count, 3)+1)) if count > 1 else [1]
            arr = ds.read(bands, masked=True).astype(np.float32)
            if np.ma.isMaskedArray(arr): arr = arr.filled(np.nan)
            meta.update({
                "sensor_name": meta.get("sensor_name") if meta.get("sensor_name") != "UNKNOWN" else "UNKNOWN",
                "width": ds.width, "height": ds.height,
                "count": count, "dtype": str(ds.dtypes[0]),
                "resolution_m_per_px": float(abs(ds.transform.a)) if ds.transform else None,
                "crs": str(ds.crs) if ds.crs else None,
                "geotransform": tuple(ds.transform) if ds.transform else None,
                "bounds": tuple(ds.bounds),
                "driver": ds.driver,
                "tags": ds.tags(),
            })
            # Multiband composite: robust percentile-normalized average.
            comps = [_norm01(b) for b in arr]
            gray = np.nanmean(np.stack(comps), axis=0)
            gray = (255*np.nan_to_num(gray)).astype(np.uint8)
    else:
        gray = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if gray is None: raise FileNotFoundError(path)
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        elif gray.ndim == 2 and gray.dtype != np.uint8:
            gray = (255*_norm01(gray)).astype(np.uint8)
        else:
            gray = gray.astype(np.uint8)
        meta.update({"width": gray.shape[1], "height": gray.shape[0]})
    if illumination_normalize:
        gray = _illumination_normalize(gray)
    return ImageRecord(gray, meta, path)
