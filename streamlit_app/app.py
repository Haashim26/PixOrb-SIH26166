# =======================================================================
# 6. Pipeline configuration / backend
# =======================================================================

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
    st.markdown(
        "### Available"
        if metrics.get("learned_available")
        else "### Unavailable"
    )
