"""
PixOrb — SIH26166 Streamlit application.

Thin UI wrapper around the same pixorb registration pipeline
used by the desktop application and FastAPI web application.

The registration algorithm itself is NOT duplicated here.

This file handles:
    - image upload
    - pipeline execution
    - evaluation metric presentation
    - visual result presentation
    - artifact downloads
"""

import sys
import tempfile
import uuid
from pathlib import Path

import streamlit as st


# ============================================================================
# REPOSITORY IMPORT
# ============================================================================

# Make the repository root importable so `import pixorb` works correctly
# on Streamlit Cloud.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pixorb.pipeline import run  # noqa: E402


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="PixOrb — Lunar Image Registration",
    page_icon="🌙",
    layout="wide",
)


# ============================================================================
# HEADER
# ============================================================================

st.title("🌙 PixOrb — Lunar Image Registration")

st.caption(
    "SIH26166 — Multi-modal, Sun angle and scale invariant image correspondence "
    "using Chandrayaan-2 optical images (OHRC, TMC, IIRS) against "
    "LRO NAC / SELENE references."
)


# ============================================================================
# FORMATTING HELPERS
# ============================================================================

def number(value, digits=3):
    """Safely format a numeric value."""
    if value is None:
        return "—"

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def integer(value):
    """Safely format an integer."""
    if value is None:
        return "—"

    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return str(value)


def percentage(value, digits=2):
    """Format a ratio such as 0.9792 as a percentage."""
    if value is None:
        return "—"

    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def text_value(value):
    """Safely convert a value to display text."""
    if value is None:
        return "—"

    if isinstance(value, bool):
        return "Yes" if value else "No"

    return str(value)


def metric_card(label, value, help_text=None):
    """Display a standard Streamlit metric card."""
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


# ============================================================================
# INPUT IMAGES
# ============================================================================

st.subheader("Input images")

input_col1, input_col2 = st.columns(2)

with input_col1:

    reference_file = st.file_uploader(
        "Reference image (fixed)",
        type=[
            "jpg",
            "jpeg",
            "png",
            "tif",
            "tiff",
        ],
        help=(
            "Example: LRO NAC or SELENE reference image. "
            "The reference image remains fixed during registration."
        ),
    )


with input_col2:

    source_file = st.file_uploader(
        "Source image (moving)",
        type=[
            "jpg",
            "jpeg",
            "png",
            "tif",
            "tiff",
        ],
        help=(
            "Example: Chandrayaan-2 OHRC, TMC-2 or IIRS image. "
            "The source image is registered to the reference."
        ),
    )


# ============================================================================
# RUN REGISTRATION
# ============================================================================

run_clicked = st.button(
    "🚀 Run registration",
    type="primary",
    use_container_width=True,
    disabled=not (
        reference_file
        and source_file
    ),
)


if run_clicked:

    with st.spinner(
        "Running PixOrb: coarse alignment → learned matching → "
        "geometric verification → fine refinement..."
    ):

        # ---------------------------------------------------------------
        # Create a unique temporary workspace for this registration.
        # ---------------------------------------------------------------

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

        # ---------------------------------------------------------------
        # Preserve uploaded file extensions.
        # ---------------------------------------------------------------

        ref_suffix = Path(
            reference_file.name
        ).suffix.lower()

        src_suffix = Path(
            source_file.name
        ).suffix.lower()

        ref_path = (
            input_dir
            / f"reference{ref_suffix}"
        )

        src_path = (
            input_dir
            / f"source{src_suffix}"
        )

        ref_path.write_bytes(
            reference_file.getvalue()
        )

        src_path.write_bytes(
            source_file.getvalue()
        )

        try:

            # -----------------------------------------------------------
            # IMPORTANT:
            #
            # The actual PixOrb algorithm is executed here.
            #
            # This Streamlit application does NOT duplicate or modify
            # the registration algorithm.
            # -----------------------------------------------------------

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

            with st.expander(
                "Technical error"
            ):
                st.exception(e)

            st.session_state.pop(
                "last_result",
                None,
            )

            st.session_state.pop(
                "last_results_dir",
                None,
            )


# ============================================================================
# RESULTS
# ============================================================================

if "last_result" in st.session_state:

    result = st.session_state[
        "last_result"
    ]

    results_dir = Path(
        st.session_state[
            "last_results_dir"
        ]
    )

    metrics = result.get(
        "metrics",
        {},
    )


    # ========================================================================
    # SUCCESS
    # ========================================================================

    st.success(
        "Registration complete — PixOrb successfully processed the image pair."
    )


    # ========================================================================
    # EVALUATION METRICS
    # ========================================================================

    st.subheader(
        "Evaluation metrics"
    )

    st.caption(
        "Registration accuracy, correspondence quality and spatial coverage "
        "computed directly by the PixOrb pipeline."
    )


    # ------------------------------------------------------------------------
    # 1. PRIMARY REGISTRATION QUALITY
    # ------------------------------------------------------------------------

    st.markdown(
        "#### Registration quality"
    )

    primary_col1, primary_col2, primary_col3 = st.columns(3)

    with primary_col1:

        metric_card(
            "Geometric RMSE",
            f"{number(metrics.get('rmse_px'), 3)} px",
            (
                "Root mean square geometric residual of "
                "the final verified correspondences."
            ),
        )

    with primary_col2:

        metric_card(
            "Mean reprojection error",
            f"{number(metrics.get('mean_reprojection_error_px'), 3)} px",
            (
                "Mean geometric reprojection error after "
                "the final transformation."
            ),
        )

    with primary_col3:

        metric_card(
            "Final RANSAC inlier ratio",
            percentage(
                metrics.get(
                    "final_ransac_inlier_ratio"
                )
            ),
            (
                "Percentage of final correspondences "
                "that remain geometrically consistent."
            ),
        )


    st.markdown("")


    # ------------------------------------------------------------------------
    # 2. CORRESPONDENCE SUMMARY
    # ------------------------------------------------------------------------

    st.markdown(
        "#### Correspondence summary"
    )

    match_col1, match_col2, match_col3 = st.columns(3)

    with match_col1:

        metric_card(
            "Candidate correspondences",
            integer(
                metrics.get(
                    "candidate_matches"
                )
            ),
            (
                "Learned feature correspondences "
                "before final geometric filtering."
            ),
        )

    with match_col2:

        metric_card(
            "Final correspondences",
            integer(
                metrics.get(
                    "final_match_count"
                )
            ),
            (
                "Correspondences retained after "
                "final verification."
            ),
        )

    with match_col3:

        metric_card(
            "Retained from candidates",
            percentage(
                metrics.get(
                    "overall_retained_ratio"
                )
            ),
            (
                "Final retained correspondences "
                "relative to candidate correspondences."
            ),
        )


    st.markdown("")


    # ------------------------------------------------------------------------
    # 3. GEOMETRIC VERIFICATION
    # ------------------------------------------------------------------------

    st.markdown(
        "#### Geometric verification"
    )

    verification_col1, verification_col2 = st.columns(2)

    # ------------------------------------------------------------------------
    # RANSAC
    # ------------------------------------------------------------------------

    with verification_col1:

        st.markdown(
            "**RANSAC verification**"
        )

        st.write(
            f"Initial RANSAC inliers: "
            f"**{integer(metrics.get('initial_ransac_inlier_count'))}**"
        )

        st.write(
            f"Initial inlier ratio: "
            f"**{percentage(metrics.get('initial_ransac_inlier_ratio'))}**"
        )

        st.write(
            f"Final RANSAC inliers: "
            f"**{integer(metrics.get('final_ransac_inlier_count'))}**"
        )

        st.write(
            f"Final inlier ratio: "
            f"**{percentage(metrics.get('final_ransac_inlier_ratio'))}**"
        )


    # ------------------------------------------------------------------------
    # REPROJECTION ERROR
    # ------------------------------------------------------------------------

    with verification_col2:

        st.markdown(
            "**Reprojection error**"
        )

        st.write(
            f"Mean: "
            f"**{number(metrics.get('mean_reprojection_error_px'), 3)} px**"
        )

        st.write(
            f"Median: "
            f"**{number(metrics.get('median_reprojection_error_px'), 3)} px**"
        )

        st.write(
            f"Maximum: "
            f"**{number(metrics.get('max_reprojection_error_px'), 3)} px**"
        )

        st.write(
            f"RMSE: "
            f"**{number(metrics.get('rmse_px'), 3)} px**"
        )


    st.markdown("")


    # ------------------------------------------------------------------------
    # 4. SPATIAL COVERAGE
    # ------------------------------------------------------------------------

    st.markdown(
        "#### Spatial coverage"
    )

    spatial_col1, spatial_col2 = st.columns(2)

    with spatial_col1:

        metric_card(
            "Grid occupancy",
            percentage(
                metrics.get(
                    "grid_occupancy_ratio"
                )
            ),
            (
                "Fraction of the spatial grid occupied "
                "by final correspondences."
            ),
        )

    with spatial_col2:

        metric_card(
            "Spatial entropy",
            number(
                metrics.get(
                    "spatial_entropy_normalized"
                ),
                3,
            ),
            (
                "Normalized spatial distribution entropy."
            ),
        )


    spatial_detail_col1, spatial_detail_col2 = st.columns(2)

    with spatial_detail_col1:

        st.write(
            f"**Occupied cells:** "
            f"{integer(metrics.get('occupied_cells'))}"
        )

    with spatial_detail_col2:

        st.write(
            f"**Total grid cells:** "
            f"{integer(metrics.get('total_cells'))}"
        )


    st.markdown("")


    # ------------------------------------------------------------------------
    # 5. IMAGE OVERLAP
    # ------------------------------------------------------------------------

    st.markdown(
        "#### Image overlap"
    )

    overlap_col1, overlap_col2, overlap_col3 = st.columns(3)

    with overlap_col1:

        metric_card(
            "Overlap",
            f"{number(metrics.get('overlap_percent'), 2)}%",
        )

    with overlap_col2:

        metric_card(
            "Overlap MAE",
            number(
                metrics.get(
                    "overlap_mae"
                ),
                3,
            ),
        )

    with overlap_col3:

        metric_card(
            "Overlap RMSE",
            number(
                metrics.get(
                    "overlap_rmse"
                ),
                3,
            ),
        )


    # ------------------------------------------------------------------------
    # 6. PIXORB CONFIGURATION
    # ------------------------------------------------------------------------

    st.markdown(
        "#### PixOrb configuration"
    )

    config_col1, config_col2 = st.columns(2)

    with config_col1:

        st.markdown(
            "**Feature backend**"
        )

        st.write(
            text_value(
                metrics.get(
                    "backend"
                )
            )
        )

        st.markdown(
            "**Geometric model**"
        )

        st.write(
            text_value(
                metrics.get(
                    "model"
                )
            )
        )


    with config_col2:

        st.markdown(
            "**RANSAC threshold**"
        )

        st.write(
            f"{number(metrics.get('ransac_threshold_px'), 2)} px"
        )

        st.markdown(
            "**Learned matching**"
        )

        if metrics.get(
            "learned_available"
        ):
            st.write(
                "Available"
            )
        else:
            st.write(
                "Unavailable"
            )


    # ------------------------------------------------------------------------
    # 7. ADVANCED COARSE ALIGNMENT
    # ------------------------------------------------------------------------

    coarse = metrics.get(
        "coarse_alignment",
        {},
    )

    if isinstance(
        coarse,
        dict,
    ):

        with st.expander(
            "Coarse alignment details",
            expanded=False,
        ):

            st.caption(
                "Initial scale, rotation and translation estimated "
                "before learned correspondence matching."
            )

            coarse_col1, coarse_col2 = st.columns(2)

            with coarse_col1:

                st.write(
                    f"**Estimated scale:** "
                    f"{number(coarse.get('scale'), 4)}"
                )

                st.write(
                    f"**Rotation:** "
                    f"{number(coarse.get('rotation_deg'), 3)}°"
                )

                st.write(
                    f"**Translation X:** "
                    f"{number(coarse.get('translation_dx'), 3)} px"
                )

                st.write(
                    f"**Translation Y:** "
                    f"{number(coarse.get('translation_dy'), 3)} px"
                )

            with coarse_col2:

                st.write(
                    f"**Scale confidence:** "
                    f"{number(coarse.get('scale_confidence'), 3)}"
                )

                st.write(
                    f"**Translation confidence:** "
                    f"{number(coarse.get('translation_confidence'), 3)}"
                )

                st.write(
                    f"**Pyramid levels:** "
                    f"{integer(coarse.get('pyramid_levels'))}"
                )

                st.write(
                    f"**Estimation level:** "
                    f"{integer(coarse.get('estimation_level'))}"
                )


    # ========================================================================
    # FULL METRICS JSON
    # ========================================================================

    with st.expander(
        "Full metrics JSON",
        expanded=False,
    ):

        st.json(
            metrics
        )


    # ========================================================================
    # VISUAL RESULTS
    # ========================================================================

    st.subheader(
        "Visual results"
    )

    st.caption(
        "The following outputs are generated directly by the PixOrb "
        "registration pipeline."
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


    # Display three images per row.
    for row_start in range(
        0,
        len(image_labels),
        3,
    ):

        row_items = image_labels[
            row_start:
            row_start + 3
        ]

        img_cols = st.columns(3)

        for col, (fname, label) in zip(
            img_cols,
            row_items,
        ):

            fpath = (
                results_dir
                / fname
            )

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


    # ========================================================================
    # DOWNLOAD ARTIFACTS
    # ========================================================================

    st.subheader(
        "Download artifacts"
    )

    st.caption(
        "Download the registered image, correspondence data and "
        "evaluation metrics."
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

        fpath = (
            results_dir
            / fname
        )

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


    # ========================================================================
    # TECHNICAL PIPELINE SUMMARY
    # ========================================================================

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
