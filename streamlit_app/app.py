"""
PixOrb — SIH26166 Streamlit demo.

A thin UI wrapper around the exact same `pixorb` registration pipeline used
by the desktop app and the FastAPI web app. No duplicate algorithm logic —
this file only handles uploads, calls pixorb.pipeline.run(), and displays
the results.
"""
import sys
import tempfile
import uuid
from pathlib import Path

import streamlit as st

# Make the repo root (one level up) importable so `import pixorb` works
# regardless of Streamlit Cloud's working directory.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pixorb.pipeline import run  # noqa: E402

st.set_page_config(page_title="PixOrb — Lunar Image Registration", layout="wide")

st.title("🌙 PixOrb — Lunar Image Registration")
st.caption(
    "SIH26166 — Multi-modal, Sun angle and scale invariant image correspondence "
    "using Chandrayaan-2 optical images (OHRC, TMC, IIRS) against LRO NAC / SELENE references."
)

col1, col2 = st.columns(2)
with col1:
    reference_file = st.file_uploader(
        "Reference image (fixed) — e.g. LRO NAC / SELENE",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
    )
with col2:
    source_file = st.file_uploader(
        "Source image (moving) — e.g. Chandrayaan-2 OHRC / TMC-2 / IIRS",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
    )

run_clicked = st.button("Run registration", type="primary", disabled=not (reference_file and source_file))

if run_clicked:
    with st.spinner("Running the full pipeline (coarse align → learned matching → RANSAC → sub-pixel refine)..."):
        run_id = uuid.uuid4().hex[:12]
        work_dir = Path(tempfile.gettempdir()) / "pixorb_streamlit_runs" / run_id
        input_dir = work_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        ref_path = input_dir / f"reference{Path(reference_file.name).suffix.lower()}"
        src_path = input_dir / f"source{Path(source_file.name).suffix.lower()}"
        ref_path.write_bytes(reference_file.getvalue())
        src_path.write_bytes(source_file.getvalue())

        try:
            result = run(ref_path, src_path, work_dir / "results")
            st.session_state["last_result"] = result
            st.session_state["last_results_dir"] = work_dir / "results"
        except Exception as e:
            st.error(f"Registration failed: {e}")
            st.session_state.pop("last_result", None)

if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    results_dir = st.session_state["last_results_dir"]
    metrics = result.get("metrics", {})

    st.success("Registration complete.")

    st.subheader("Evaluation metrics")
    metric_cols = st.columns(len(metrics) if metrics else 1)
    for (k, v), c in zip(metrics.items(), metric_cols):
        c.metric(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    with st.expander("Full metrics JSON"):
        st.json(metrics)

    st.subheader("Visual results")
    img_cols = st.columns(3)
    image_labels = [
        ("reference.png", "Reference"),
        ("source_original.png", "Source (original)"),
        ("matches.png", "Matched keypoints"),
        ("registered.png", "Registered source"),
        ("overlay.png", "Overlay (reference + registered)"),
        ("difference_map.png", "Difference map"),
    ]
    for i, (fname, label) in enumerate(image_labels):
        fpath = results_dir / fname
        if fpath.exists():
            with img_cols[i % 3]:
                st.image(str(fpath), caption=label, use_container_width=True)

    st.subheader("Download artifacts")
    dl_cols = st.columns(4)
    downloads = [
        ("registered.png", "Registered image"),
        ("match_points.csv", "Match points (CSV)"),
        ("match_points.json", "Match points (JSON)"),
        ("metrics.json", "Metrics (JSON)"),
    ]
    for i, (fname, label) in enumerate(downloads):
        fpath = results_dir / fname
        if fpath.exists():
            with dl_cols[i % 4]:
                st.download_button(label, data=fpath.read_bytes(), file_name=fname)
