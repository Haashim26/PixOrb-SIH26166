const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

/* Main tabs */
$$(".tab").forEach((button) => {
    button.onclick = () => {
        $$(".tab").forEach((x) => x.classList.remove("active"));
        $$(".panel").forEach((x) => x.classList.remove("active"));

        button.classList.add("active");

        const panel = $("#" + button.dataset.tab);
        if (panel) panel.classList.add("active");
    };
});

/* Registered-output subtabs */
$$(".ot").forEach((button) => {
    button.onclick = () => {
        $$(".ot").forEach((x) => x.classList.remove("active"));
        $$(".out").forEach((x) => x.classList.remove("active"));

        button.classList.add("active");

        const output = $("#" + button.dataset.o);
        if (output) output.classList.add("active");
    };
});

/* Run registration */
$("#run").onclick = async () => {
    const reference = $("#ref").files[0];
    const source = $("#src").files[0];

    if (!reference || !source) {
        $("#status").textContent = "Select both images first.";
        return;
    }

    const formData = new FormData();
    formData.append("reference", reference);
    formData.append("source", source);

    $("#run").disabled = true;
    $("#status").textContent =
        "Running PixOrb coarse-to-fine registration…";

    try {
        const response = await fetch("/api/register", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Registration failed");
        }

        $("#results").classList.remove("hidden");

        for (const [key, value] of Object.entries(data.images || {})) {
            const image = $("#" + key);

            if (image && value) {
                image.src = value + "?t=" + Date.now();
            }
        }

        renderMetrics(data.metrics || {});

        $("#status").textContent =
            "Registration complete — " +
            (data.metrics?.backend || "unknown backend");

    } catch (error) {
        $("#status").textContent =
            error?.message || "Registration failed.";
    } finally {
        $("#run").disabled = false;
    }
};

/* Judge-facing metric cards */
function renderMetrics(m) {
    const items = [
        ["RMSE", fmt(m.rmse_px) + " px"],
        ["Mean reprojection error", fmt(m.mean_reprojection_error_px) + " px"],
        ["Final matches", integer(m.final_match_count)],
        ["Final RANSAC verification", pct(m.final_ransac_inlier_ratio)],
        ["Initial RANSAC inliers", integer(m.initial_ransac_inlier_count)],
        ["Candidate matches", integer(m.candidate_matches)],
        ["Retained from candidates", pct(m.overall_retained_ratio)],
        ["Spatial entropy", fmt(m.spatial_entropy_normalized)]
    ];

    $("#cards").innerHTML = items
        .map(
            ([label, value]) => `
                <div class="card">
                    <div class="k">${label}</div>
                    <div class="v">${value}</div>
                </div>
            `
        )
        .join("");

    $("#json").textContent = JSON.stringify(m, null, 2);
}

function fmt(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n.toFixed(3) : "—";
}

function integer(value) {
    const n = Number(value);
    return Number.isFinite(n) ? Math.round(n).toString() : "—";
}

function pct(value) {
    const n = Number(value);

    if (!Number.isFinite(n)) return "—";

    return (n * 100).toFixed(2) + "%";
}