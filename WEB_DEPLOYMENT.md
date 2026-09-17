# PixOrb — shareable web demo

The web interface is a thin front-end over the exact same `pixorb` registration core used by the desktop application. It is not a second or simplified algorithm.

## Local web demo

From the project root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-web.txt
python -m uvicorn web.main:app --host 127.0.0.1 --port 7860
```

Open `http://127.0.0.1:7860`.

## Public team/judge URL

For a click-to-open demo, deploy this project to a Python-capable host (for example a Hugging Face Docker Space). The host must have enough RAM/CPU for PyTorch + SuperPoint/LightGlue and must expose port 7860.

The included `Dockerfile` starts the same web application automatically.

## Important

- No Streamlit is used.
- The web UI uses the same PixOrb pipeline as the desktop product.
- Uploads are processed on the server running PixOrb.
- The first learned-model execution may need to obtain pretrained weights. After the weights are cached, inference can run without downloading them again.
- The demo accepts JPG/JPEG/PNG/TIFF/GeoTIFF/COG inputs up to 30 MB per image by default.
