"""PixOrb Windows installation check."""
import importlib
import sys

modules = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("cv2", "cv2"),
    ("rasterio", "rasterio"),
    ("PIL", "PIL"),
    ("matplotlib", "matplotlib"),
    ("PySide6", "PySide6"),
    ("torch", "torch"),
    ("lightglue", "lightglue"),
]

failed = False
for label, mod in modules:
    try:
        m = importlib.import_module(mod)
        version = getattr(m, "__version__", "OK")
        print(f"{label}: OK {version}")
    except Exception as exc:
        failed = True
        print(f"{label}: FAIL — {exc}")

try:
    import torch
    print(f"torch CUDA available: {torch.cuda.is_available()}")
except Exception:
    pass

try:
    from lightglue import LightGlue, SuperPoint
    print("SuperPoint + LightGlue: OK")
except Exception as exc:
    failed = True
    print(f"SuperPoint + LightGlue: FAIL — {exc}")

print(f"Python: {sys.version}")
sys.exit(1 if failed else 0)
