"""
PixOrb — SIH26166 Streamlit application.

Thin UI wrapper around the same pixorb registration pipeline
used by the desktop application and FastAPI web application.

The registration algorithm itself is NOT duplicated here.
"""

import sys
import tempfile
import uuid
from pathlib import Path

import streamlit as st


# ---------------------------------------------------------------------------
# Repository import
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pixorb.pipeline import run  # noqa: E402


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PixOrb — Lunar Image Registration",
    page_icon="🌙",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🌙 PixOrb — Lunar Image Registration")

st.caption(
    "SIH26166 — Multi-modal, Sun angle and scale invariant image correspondence "
    "using Chandrayaan-2 optical images (OHRC, TMC, IIRS) against "
    "LRO NAC / SELENE references."
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def number(value, digits=3):
    if value is None:
        return "—"

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def integer(value):
    if value is None:
        return "—"

    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return str(value)


def percentage(value, digits=2):
    if value is None:
        return "—"

    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def text_value(value):
    if value is None:
        return "—"

    if isinstance(value, bool):
        return "Yes" if value else "No"

    return str(value)


def metric_card(label, value, help_text=None):
    if help_text:
        st.metric(
            label=label,
            value=value,
            help=help_text,
        )
    else:
        st.metric(
            label=label,
            value=value,
        )


# ---------------------------------------------------------------------------
# Input images
# ---------------------------------------------------------------------------

st.subheader("Input images")

input_col1, input_col2 = st.columns(2)

with input_col1:
    reference_file = st.file_uploader(
        "Reference image (fixed)",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        help=(
            "Example: LRO NAC or SELENE reference image. "
            "The reference image remains fixed during registration."
        ),
    )

with input_col2:
    source_file = st.file_uploader(
        "Source image (moving)",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        help=(
            "Example: Chandrayaan-2 OHRC, TMC-2 or IIRS image. "
            "The source image is registered to the reference."
        ),
    )


# ---------------------------------------------------------------------------
# Run registration
# ---------------------------------------------------------------------------

run_clicked = st.button(
    "🚀 Run registration",
    type="primary",
    use_container_width=True,
    disabled=not (reference_file and source_file),
)


if run_clicked:

    with st.spinner(
        "Running PixOrb: coarse alignment → learned matching → "
        "geometric verification → fine refinement..."
    ):

        run_id = uuid.uuid4().hex[:12]

        work_dir = (
            Path(tempfile.gettempdir())
            / "pixorb_streamlit_runs"
            / run_id
        )

        input_dir = work_dir / "input"

        input_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        ref_suffix = Path(
            reference_file.name
        ).suffix.lower()

        src_suffix = Path(
            source_file.name
        ).suffix.lower()

        ref_path = input_dir / f"reference{ref_suffix}"
        src_path = input_dir / f"source{src_suffix}"

        ref_path.write_bytes(
            reference_file.getvalue()
        )

        src_path.write_bytes(
            source_file.getvalue()
        )

        try:

            # IMPORTANT:
            # This calls the original PixOrb pipeline.
            result = run(
                ref_path,
                src_path,
                work_dir / "results",
            )

            st.session_state["last_result"] = result

            st.session_state["last_results_dir"] = (
                work_dir / "results"
            )

        except Exception as e:

            st.error(
                "Registration failed. "
                "Please check the uploaded images and try again."
            )

            with st.expander("Technical error"):
                st.exception(e)

            st.session_state.pop(
                "last_result",
                None,
            )

            st.session_state.pop(
                "last_results_dir",
                None,
            )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if "last_result" in st.session_state:

    result = st.session_state["last_result"]

    results_dir = Path(
        st.session_state["last_results_dir"]
    )

    metrics = result.get(
        "metrics",
        {},
    )


    # =======================================================================
    # Success
    # =======================================================================

    st.success(
        "Registration complete — PixOrb successfully processed the image pair."
    )


    # =======================================================================
    # Evaluation metrics
    # =======================================================================

    st.subheader("Evaluation metrics")

    st.caption(
        "Geometric registration quality and matching statistics produced "
        "directly by the PixOrb pipeline."
    )


    # -----------------------------------------------------------------------
    # Registration quality
    # -----------------------------------------------------------------------

    st.markdown("#### Registration quality")

    quality_cols = st.columns(4)

    with quality_cols[0]:
        metric_card(
            "RMSE",
            f"{number(metrics.get('rmse_px'), 3)} px",
            (
                "Root mean square geometric residual of the "
                "final verified matches."
            ),
        )

    with quality_cols[1]:
        metric_card(
            "Mean reprojection error",
            f"{number(metrics.get('mean_reprojection_error_px'), 3)} px",
            (
                "Mean geometric reprojection error after "
                "the final transform."
            ),
        )

    with quality_cols[2]:
        metric_card(
            "Median reprojection error",
            f"{number(metrics.get('median_reprojection_error_px'), 3)} px",
            (
                "Median geometric reprojection error of "
                "the final matches."
            ),
        )

    with quality_cols[3]:
        metric_card(
            "Max reprojection error",
            f"{number(metrics.get('max_reprojection_error_px'), 3)} px",
            (
                "Largest geometric reprojection residual "
                "among final matches."
            ),
        )


    # -----------------------------------------------------------------------
    # Matching and geometric verification
    # -----------------------------------------------------------------------

    st.markdown("#### Matching and geometric verification")

    matching_cols = st.columns(4)

    with matching_cols[0]:
        metric_card(
            "Candidate matches",
            integer(
                metrics.get("candidate_matches")
            ),
            (
                "Learned feature correspondences before "
                "final geometric filtering."
            ),
        )

    with matching_cols[1]:
        metric_card(
            "Final matches",
            integer(
                metrics.get("final_match_count")
            ),
            (
                "Correspondences retained after "
                "the final verification stage."
            ),
        )

    with matching_cols[2]:
        metric_card(
            "RANSAC inlier ratio",
            percentage(
                metrics.get("final_ransac_inlier_ratio")
            ),
            (
                "Final geometrically consistent matches "
                "divided by final matches."
            ),
        )

    with matching_cols[3]:
        metric_card(
            "Overall retained",
            percentage(
                metrics.get("overall_retained_ratio")
            ),
            (
                "Final retained matches relative to "
                "candidate matches."
            ),
        )


    # -----------------------------------------------------------------------
    # RANSAC verification
    # -----------------------------------------------------------------------

    st.markdown("#### RANSAC verification")

    ransac_cols = st.columns(4)

    with ransac_cols[0]:
        metric_card(
            "Initial RANSAC inliers",
            integer(
                metrics.get("initial_ransac_inlier_count")
            ),
        )

    with ransac_cols[1]:
        metric_card(
            "Initial inlier ratio",
            percentage(
                metrics.get("initial_ransac_inlier_ratio")
            ),
        )

    with ransac_cols[2]:
        metric_card(
            "Final RANSAC inliers",
            integer(
                metrics.get("final_ransac_inlier_count")
            ),
        )

    with ransac_cols[3]:
        metric_card(
            "Final inlier ratio",
            percentage(
                metrics.get("final_ransac_inlier_ratio")
            ),
        )


    # -----------------------------------------------------------------------
    # Spatial coverage
    # -----------------------------------------------------------------------

    st.markdown("#### Spatial coverage")

    spatial_cols = st.columns(4)

    with spatial_cols[0]:
        metric_card(
            "Grid occupancy",
            percentage(
                metrics.get("grid_occupancy_ratio")
            ),
            (
                "Fraction of the spatial grid occupied "
                "by final matches."
            ),
        )

    with spatial_cols[1]:
        metric_card(
            "Spatial entropy",
            number(
                metrics.get("spatial_entropy_normalized"),
                3,
            ),
            (
                "Normalized spatial distribution entropy."
            ),
        )

    with spatial_cols[2]:
        metric_card(
            "Occupied cells",
            integer(
                metrics.get("occupied_cells")
            ),
        )

    with spatial_cols[3]:
        metric_card(
            "Total grid cells",
            integer(
                metrics.get("total_cells")
            ),
        )


    # -----------------------------------------------------------------------
    # Image overlap
    # -----------------------------------------------------------------------

    st.markdown("#### Image overlap")

    overlap_cols = st.columns(4)

    with overlap_cols[0]:
        metric_card(
            "Overlap",
            f"{number(metrics.get('overlap_percent'), 2)}%",
        )

    with overlap_cols[1]:
        metric_card(
            "Overlap MAE",
            number(
                metrics.get("overlap_mae"),
                3,
            ),
        )

    with overlap_cols[2]:
        metric_card(
            "Overlap RMSE",
            number(
                metrics.get("overlap_rmse"),
                3,
            ),
        )

    with overlap_cols[3]:
        metric_card(
            "Sub-pixel updates",
            integer(
                metrics.get("subpixel_changed_count")
            ),
        )


    # -----------------------------------------------------------------------
    # PixOrb pipeline
    #
    # Long text values are deliberately NOT displayed using st.metric()
    # because Streamlit truncates them.
    # -----------------------------------------------------------------------

    st.markdown("#### PixOrb pipeline")

    pipeline_cols = st.columns(4)

    with pipeline_cols[0]:

        st.markdown("**Feature backend**")

        st.markdown(
            f"### {text_value(metrics.get('backend'))}"
        )

    with pipeline_cols[1]:

        st.markdown("**Geometric model**")

        st.markdown(
            f"### {text_value(metrics.get('model'))}"
        )

    with pipeline_cols[2]:

        st.markdown("**RANSAC threshold**")

        st.markdown(
            f"### {number(metrics.get('ransac_threshold_px'), 2)} px"
        )

    with pipeline_cols[3]:

        st.markdown("**Learned matching**")

        if metrics.get("learned_available"):
            st.markdown("### Available")
        else:
            st.markdown("### Unavailable")


    # -----------------------------------------------------------------------
    # Coarse alignment
    # -----------------------------------------------------------------------

    coarse = metrics.get(
        "coarse_alignment",
        {},
    )

    if isinstance(coarse, dict):

        st.markdown("#### Coarse alignment")

        coarse_cols = st.columns(4)

        with coarse_cols[0]:
            metric_card(
                "Estimated scale",
                number(
                    coarse.get("scale"),
                    4,
                ),
            )

        with coarse_cols[1]:
            metric_card(
                "Rotation",
                f"{number(coarse.get('rotation_deg'), 3)}°",
            )

        with coarse_cols[2]:
            metric_card(
                "Translation X",
                f"{number(coarse.get('translation_dx'), 3)} px",
            )

        with coarse_cols[3]:
            metric_card(
                "Translation Y",
                f"{number(coarse.get('translation_dy'), 3)} px",
            )


        confidence_cols = st.columns(4)

        with confidence_cols[0]:
            metric_card(
                "Scale confidence",
                number(
                    coarse.get("scale_confidence"),
                    3,
                ),
            )

        with confidence_cols[1]:
            metric_card(
                "Translation confidence",
                number(
                    coarse.get("translation_confidence"),
                    3,
                ),
            )

        with confidence_cols[2]:
            metric_card(
                "Pyramid levels",
                integer(
                    coarse.get("pyramid_levels")
                ),
            )

        with confidence_cols[3]:
            metric_card(
                "Estimation level",
                integer(
                    coarse.get("estimation_level")
                ),
            )


    # =======================================================================
    # Full metrics JSON
    # =======================================================================

    with st.expander(
        "Full metrics JSON",
        expanded=False,
    ):
        st.json(metrics)


    # =======================================================================
    # Visual results
    # =======================================================================

    st.subheader("Visual results")

    st.caption(
        "The following outputs are generated by the PixOrb registration pipeline."
    )

    image_labels = [
        (
            "reference.png",
            "Reference",
        ),
        (
            "source_original.png",
            "Source (original)",
        ),
        (
            "matches.png",
            "Matched keypoints",
        ),
        (
            "registered.png",
            "Registered source",
        ),
        (
            "overlay.png",
            "Overlay — reference + registered",
        ),
        (
            "difference_map.png",
            "Difference map",
        ),
    ]


    for row_start in range(
        0,
        len(image_labels),
        3,
    ):

        row_items = image_labels[
            row_start:row_start + 3
        ]

        img_cols = st.columns(3)

        for col, (fname, label) in zip(
            img_cols,
            row_items,
        ):

            fpath = results_dir / fname

            with col:

                if fpath.exists():

                    st.image(
                        str(fpath),
                        caption=label,
                        use_container_width=True,
                    )

                else:

                    st.warning(
                        f"{label} output was not generated."
                    )


    # =======================================================================
    # Download artifacts
    # =======================================================================

    st.subheader("Download artifacts")

    st.caption(
        "Download the registered image, correspondence data and evaluation metrics."
    )

    downloads = [
        (
            "registered.png",
            "⬇️ Registered image",
        ),
        (
            "match_points.csv",
            "⬇️ Match points — CSV",
        ),
        (
            "match_points.json",
            "⬇️ Match points — JSON",
        ),
        (
            "metrics.json",
            "⬇️ Metrics — JSON",
        ),
    ]


    download_cols = st.columns(4)

    for col, (fname, label) in zip(
        download_cols,
        downloads,
    ):

        fpath = results_dir / fname

        with col:

            if fpath.exists():

                st.download_button(
                    label=label,
                    data=fpath.read_bytes(),
                    file_name=fname,
                    use_container_width=True,
                )

            else:

                st.caption(
                    f"{fname} not available."
                )


    # =======================================================================
    # Technical pipeline summary
    # =======================================================================

    with st.expander(
        "PixOrb processing pipeline"
    ):

        st.markdown(
            """
**Input**

Reference image + source image

↓

**Data ingestion / preprocessing**

Image loading and preprocessing

↓

**Coarse alignment**

Multi-resolution image pyramid and phase-correlation based alignment

↓

**Learned feature matching**

SuperPoint + LightGlue

↓

**Geometric verification**

Homography / affine estimation with RANSAC

↓

**Fine refinement**

NCC-based refinement, sub-pixel localization and spatial filtering

↓

**Output**

Registered image, correspondence points, transformation matrix and
evaluation metrics
"""
        )
