# PixOrb Web Demo

FastAPI web front-end over the same `pixorb` core used by the desktop product.

Run from the project root:

```bash
python -m uvicorn web.main:app --host 0.0.0.0 --port 7860
```

For hosted deployment, install both `requirements.txt` and `web/requirements.txt`.
The web app does not contain a separate registration algorithm.
