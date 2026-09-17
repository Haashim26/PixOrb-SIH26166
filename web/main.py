from pathlib import Path
import json, tempfile, uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from pixorb.pipeline import run

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "web_runs"
RUNS.mkdir(exist_ok=True)
app = FastAPI(title="PixOrb — Lunar Image Registration", version="1.0")
app.mount("/static", StaticFiles(directory=ROOT / "web" / "static"), name="static")

ALLOWED = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".cog"}
MAX_BYTES = 30 * 1024 * 1024

@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")

async def save_upload(upload: UploadFile, dst: Path):
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, f"Unsupported image type: {suffix or 'unknown'}")
    total = 0
    with dst.open("wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_BYTES:
                raise HTTPException(413, "Each image must be 30 MB or smaller.")
            f.write(chunk)
    return dst

@app.post("/api/register")
async def register(reference: UploadFile = File(...), source: UploadFile = File(...)):
    run_id = uuid.uuid4().hex[:12]
    out = RUNS / run_id
    inp = out / "input"
    inp.mkdir(parents=True)
    ref_path = await save_upload(reference, inp / f"reference{Path(reference.filename or '.png').suffix.lower()}")
    src_path = await save_upload(source, inp / f"source{Path(source.filename or '.png').suffix.lower()}")
    try:
        result = run(ref_path, src_path, out / "results")
        metrics = result["metrics"]
        return {
            "run_id": run_id,
            "metrics": metrics,
            "images": {k: f"/api/runs/{run_id}/{v}" for k,v in {
                "reference":"results/reference.png",
                "source":"results/source_original.png",
                "matches":"results/matches.png",
                "registered":"results/registered.png",
                "overlay":"results/overlay.png",
                "difference":"results/difference_map.png",
                "overlap":"results/overlap_mask.png",
            }.items()},
            "files": {"metrics": f"/api/runs/{run_id}/results/metrics.json", "matches": f"/api/runs/{run_id}/results/match_points.csv", "transform": f"/api/runs/{run_id}/results/transform_source_to_reference.npy"}
        }
    except Exception as e:
        raise HTTPException(500, f"Registration failed: {e}") from e

@app.get("/api/runs/{run_id}/{path:path}")
def artifact(run_id: str, path: str):
    base = (RUNS / run_id).resolve()
    target = (base / path).resolve()
    if base not in target.parents and target != base:
        raise HTTPException(400, "Invalid path")
    if not target.is_file():
        raise HTTPException(404, "Artifact not found")
    return FileResponse(target)
