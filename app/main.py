import streamlit as st
import pandas as pd
from PIL import Image
from app.pipeline import analyze_image, get_pipeline_status

st.set_page_config(
    page_title="Military Terrain Analysis",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM UI DESIGN --------------------

st.markdown("""
<style>
.stApp {
    background: #0b1117;
    color: #e6edf3;
}
[data-testid="stSidebar"] {
    background: #0e1821;
    border-right: 1px solid #263744;
}
.block-container {
    padding-top: 1.5rem;
    max-width: 1450px;
}
h1, h2, h3 {
    color: #e6edf3 !important;
}
.hero {
    background: linear-gradient(115deg, #111d26, #0d151d);
    border: 1px solid #263744;
    border-left: 4px solid #a6b875;
    border-radius: 9px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
}
.eyebrow {
    color: #a6b875;
    font-size: .72rem;
    letter-spacing: .22em;
    font-weight: 700;
}
.metric {
    background: #111b24;
    border: 1px solid #263744;
    border-radius: 9px;
    padding: 1rem;
    min-height: 95px;
}
.metric-label {
    color: #9eacb8;
    font-size: .82rem;
}
.metric-value {
    color: #e6edf3;
    font-size: 1.55rem;
    font-weight: 750;
}
.panel {
    background: #111b24;
    border: 1px solid #263744;
    border-radius: 9px;
    padding: 1rem;
    margin-bottom: .8rem;
}
.section-title {
    color: #d2df9a;
    font-size: 1.05rem;
    font-weight: 700;
    margin: .5rem 0 .8rem;
}
div.stButton > button {
    width: 100%;
    text-align: left;
    background: #14212b;
    color: #dce6eb;
    border: 1px solid #30424e;
    border-radius: 7px;
    padding: .58rem .8rem;
    font-weight: 600;
}
div.stButton > button:hover {
    border-color: #a6b875;
    color: #d2df9a;
    background: #1a2a33;
}
div.stButton > button[kind="primary"] {
    background: #788b50;
    border-color: #9aab6b;
    color: #08110b;
}
.stFileUploader section {
    background: #111d27;
    border: 1px dashed #465966;
    border-radius: 9px;
}
hr {
    border-color: #263744;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# -------------------- NAVIGATION --------------------

PAGES = [
    "Dashboard",
    "Terrain Analysis",
    "Building Detection",
    "Model Information",
    "About Project"
]

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "analysis_error" not in st.session_state:
    st.session_state.analysis_error = None


# -------------------- HELPER FUNCTIONS --------------------

def metric(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <small>{note}</small>
        </div>
        """,
        unsafe_allow_html=True
    )


def hero(tag, title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{tag}</div>
            <h1>{title}</h1>
            <p style="color:#9eacb8">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def status_models():
    try:
        return get_pipeline_status().get("models", {})
    except Exception:
        return {}


def label_class(k):
    return {
        "urban_land": "Urban Land",
        "agriculture_land": "Agriculture Land",
        "rangeland": "Rangeland",
        "forest_land": "Forest Land",
        "water": "Water",
        "barren_land": "Barren Land"
    }.get(str(k).lower(), str(k).replace("_", " ").title())


def image_panel(img, caption):
    if img is not None:
        st.image(img, caption=caption, use_container_width=True)
    else:
        st.info(caption + " is unavailable.")


# -------------------- SIDEBAR --------------------

with st.sidebar:

    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;padding:.4rem 0 1rem">
        <div style="width:44px;height:44px;border:1px solid #788b50;
        border-radius:9px;display:grid;place-items:center;
        font-size:24px;color:#a6b875">◎</div>

        <div>
            <b style="font-size:1.1rem">MTA</b><br>
            <small style="color:#8b9aa7">Military Terrain Analysis</small>
        </div>
    </div>

    <div class="eyebrow">NAVIGATION</div>
    """, unsafe_allow_html=True)

    for p in PAGES:
        if st.button(
            ("▣  " if st.session_state.page == p else "›  ") + p,
            key="nav_" + p,
            type="primary" if st.session_state.page == p else "secondary"
        ):
            st.session_state.page = p
            st.rerun()

    st.divider()

    st.markdown(
        '<div class="eyebrow">MODEL AVAILABILITY</div>',
        unsafe_allow_html=True
    )

    statuses = status_models()

    for key, name in [
        ("deepglobe", "DeepGlobe U-Net"),
        ("spacenet", "SpaceNet U-Net"),
        ("eurosat", "EuroSAT All-Bands")
    ]:
        ready = statuses.get(key, {}).get("available", False)

        st.markdown(
            f'<p style="font-size:.82rem;color:'
            f'{"#b8d18a" if ready else "#d49b76"}">'
            f'{"●" if ready else "○"} &nbsp;{name} — '
            f'{"Ready" if ready else "Unavailable"}</p>',
            unsafe_allow_html=True
        )

    st.caption(
        "SkyView."
    )
    st.caption("Academic prototype • Local inference")


# -------------------- PAGE ROUTING --------------------

page = st.session_state.page
result = st.session_state.analysis_result


# ========================================================
#                       DASHBOARD
# ========================================================

if page == "Dashboard":

    hero(
        "COMMAND CENTRE",
        "MILITARY TERRAIN ANALYSIS SYSTEM",
        "Land-cover segmentation and experimental building detection from satellite imagery."
    )

    models = status_models()

    c1, c2, c3 = st.columns(3)

    with c1:
        metric(
            "PRIMARY MODULE",
            "DeepGlobe U-Net",
            "Land-cover segmentation"
        )

    with c2:
        metric(
            "LAND-COVER CLASSES",
            "06",
            "Urban • Agriculture • Rangeland • Forest • Water • Barren"
        )

    with c3:
        metric(
            "MODEL STATUS",
            "Ready" if models.get("deepglobe", {}).get("available") else "Check files",
            "DeepGlobe checkpoint"
        )

    st.markdown(
        '<div class="section-title">SATELLITE IMAGE ANALYSIS</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2, gap="large")

    with left:

        st.markdown(
            '<div class="panel">**01 / IMAGE INPUT**</div>',
            unsafe_allow_html=True
        )

        uploaded = st.file_uploader(
            "Upload satellite or aerial image",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            key="image_upload"
        )

        if uploaded:

            st.caption("Selected: " + uploaded.name)

            if st.button("RUN TERRAIN ANALYSIS", type="primary"):

                try:
                    with st.spinner("Running compatible models..."):

                        st.session_state.analysis_result = analyze_image(
                            uploaded,
                            run_skyview=False
                        )

                        st.session_state.analysis_error = None

                    st.rerun()

                except Exception as e:
                    st.session_state.analysis_error = str(e)

        else:
            st.caption("Supported formats: JPG, PNG, TIFF")

    with right:

        st.markdown(
            '<div class="panel">**02 / IMAGE PREVIEW**</div>',
            unsafe_allow_html=True
        )

        if result and result.get("preview") is not None:

            image_panel(
                result["preview"],
                result.get("filename", "Uploaded image")
            )

            st.caption("Input type: " + str(result.get("input_type")))

        elif uploaded and not uploaded.name.lower().endswith((".tif", ".tiff")):

            uploaded.seek(0)

            image_panel(
                Image.open(uploaded).convert("RGB"),
                "Selected image"
            )

        else:
            st.info("Uploaded image preview appears here.")

    if st.session_state.analysis_error:
        st.error("Analysis failed: " + st.session_state.analysis_error)

    if result:

        st.success(
            "Analysis completed. Open Terrain Analysis or Building Detection in the sidebar."
        )

        x, y, z = st.columns(3)

        with x:
            metric("INPUT", result.get("input_type", "Unknown"))

        with y:
            metric(
                "LAND-COVER OUTPUT",
                "Available" if result.get("deepglobe") else "Not applicable"
            )

        with z:
            metric(
                "BUILDING OUTPUT",
                "Available" if result.get("spacenet") else "Not applicable"
            )

        for warning in result.get("warnings", []):
            st.warning(warning)


# ========================================================
#                    TERRAIN ANALYSIS
# ========================================================

elif page == "Terrain Analysis":

    hero(
        "ANALYSIS MODULE 01",
        "LAND-COVER SEGMENTATION",
        "Six-class pixel-level mapping using the DeepGlobe U-Net model."
    )

    deep = result.get("deepglobe") if result else None

    if not result:

        st.info("Upload an image and run analysis from Dashboard first.")

    elif not deep:

        st.warning(
            "DeepGlobe output is unavailable for this input. "
            "RGB imagery is required; 13-band TIFF is routed to EuroSAT."
        )

    else:

        percentages = deep.get("percentages", {})

        if percentages:

            entries = sorted(
                percentages.items(),
                key=lambda x: float(x[1]),
                reverse=True
            )

            st.markdown(
                '<div class="section-title">LAND-COVER DISTRIBUTION</div>',
                unsafe_allow_html=True
            )

            cols = st.columns(3)

            for i, (k, v) in enumerate(entries):

                with cols[i % 3]:

                    metric(
                        label_class(k),
                        f"{float(v):.2f}%",
                        "Predicted pixel coverage"
                    )

            chart = pd.DataFrame(
                {"Coverage (%)": [float(v) for _, v in entries]},
                index=[label_class(k) for k, _ in entries]
            )

            st.bar_chart(chart)

        st.markdown(
            '<div class="section-title">SEGMENTATION VISUALISATION</div>',
            unsafe_allow_html=True
        )

        a, b = st.columns(2, gap="large")

        with a:

            st.markdown(
                '<div class="panel">**SEGMENTATION MASK**</div>',
                unsafe_allow_html=True
            )

            image_panel(
                deep.get("mask"),
                "Predicted land-cover mask"
            )

        with b:

            st.markdown(
                '<div class="panel">**SEGMENTATION OVERLAY**</div>',
                unsafe_allow_html=True
            )

            image_panel(
                deep.get("overlay"),
                "Land-cover overlay"
            )

        st.caption(
            "Percentages represent predicted pixel distribution, "
            "not ground-truth measurements."
        )

        st.caption(
            "Recorded validation: mIoU 64.85% • Pixel accuracy 83.38%. "
            "These do not guarantee performance on every uploaded image."
        )


# ========================================================
#                  BUILDING DETECTION
# ========================================================

elif page == "Building Detection":

    hero(
        "ANALYSIS MODULE 02",
        "BUILDING FOOTPRINT DETECTION",
        "Experimental building segmentation using the SpaceNet U-Net model."
    )

    st.warning(
        "Experimental only: recorded best validation IoU was approximately "
        "0.0954. Do not treat this as a reliable building measurement."
    )

    space = result.get("spacenet") if result else None

    if not result:

        st.info("Upload an RGB image and run analysis from Dashboard first.")

    elif not space:

        st.info(
            "SpaceNet output is unavailable for this input. "
            "13-band TIFF follows the EuroSAT route."
        )

    else:

        try:

            metric(
                "PREDICTED BUILDING COVERAGE",
                f'{float(space.get("building_percentage", 0)):.2f}%',
                "Model-predicted pixel coverage"
            )

        except (ValueError, TypeError):

            st.info("Building coverage value was not returned.")

        a, b = st.columns(2, gap="large")

        with a:

            st.markdown(
                '<div class="panel">**BUILDING MASK**</div>',
                unsafe_allow_html=True
            )

            image_panel(
                space.get("mask"),
                "SpaceNet building mask"
            )

        with b:

            st.markdown(
                '<div class="panel">**DETECTION OVERLAY**</div>',
                unsafe_allow_html=True
            )

            image_panel(
                space.get("overlay"),
                "SpaceNet overlay"
            )


# ========================================================
#                   MODEL INFORMATION
# ========================================================

elif page == "Model Information":

    hero(
        "TECHNICAL OVERVIEW",
        "MODEL INFORMATION",
        "Architectures, datasets and recorded evaluation results."
    )

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.subheader("DeepGlobe U-Net")

    st.write("**Role:** Primary land-cover segmentation")
    st.write("**Architecture:** ResNet18 encoder + U-Net decoder")
    st.write("**Dataset:** DeepGlobe Land Cover")
    st.write("**Classes:** 6")

    a, b = st.columns(2)

    with a:
        metric("BEST VALIDATION mIoU", "64.85%")

    with b:
        metric("VALIDATION PIXEL ACCURACY", "83.38%")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.subheader("SpaceNet U-Net")

    st.write("**Role:** Experimental building-footprint segmentation")
    st.write("**Architecture:** ResNet18 encoder + U-Net decoder")
    st.write("**Dataset:** SpaceNet 7")
    st.write("**Best validation IoU:** approximately 0.0954")

    st.warning(
        "Weak validation result; output is for visual demonstration only."
    )

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.subheader("EuroSAT All-Bands")

    st.write(
        "**Role:** Classification of compatible 13-band TIFF imagery"
    )
    st.write(
        "**Architecture:** ResNet18 adapted for 13 channels"
    )
    st.write(
        "**Recorded test accuracy:** 98.26% (2,711 / 2,759 samples)"
    )

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.subheader("SkyView ResNet18")

    st.write(
        "**Recorded test accuracy:** 96.42% on its own SkyView test set."
    )

    st.caption(
        "Retained in project files but excluded from primary dashboard "
        "prediction due to dataset-domain differences."
    )

    st.markdown('</div>', unsafe_allow_html=True)


# ========================================================
#                     ABOUT PROJECT
# ========================================================

elif page == "About Project":

    hero(
        "PROJECT PROFILE",
        "ABOUT MILITARY TERRAIN ANALYSIS",
        "An academic deep-learning prototype for satellite and aerial image interpretation."
    )

    st.markdown("""
    <div class="panel">
        <div class="section-title">PROJECT OBJECTIVE</div>
        Apply computer-vision models to satellite and aerial imagery
        to identify land-cover patterns and demonstrate experimental
        building-footprint segmentation.
    </div>

    <div class="panel">
        <div class="section-title">CORE CAPABILITIES</div>
        • Six-class land-cover segmentation using DeepGlobe U-Net<br>
        • Experimental building-footprint segmentation using SpaceNet U-Net<br>
        • Multispectral classification for compatible 13-band TIFF imagery using EuroSAT<br>
        • Class coverage, segmentation masks and image overlays
    </div>

    <div class="panel">
        <div class="section-title">POTENTIAL APPLICATION AREAS</div>
        • Remote-sensing education and geospatial research<br>
        • Preliminary disaster-response mapping research<br>
        • Urban and infrastructure mapping research<br>
        • Environmental and land-cover monitoring
    </div>

    <div class="panel">
        <div class="section-title">LIMITATIONS</div>
        This is a college-level prototype. Results may vary with image quality,
        geography, resolution and dataset differences. SpaceNet building
        detection has weak validation performance. Outputs are not intended
        for operational military decisions or safety-critical use.
    </div>
    """, unsafe_allow_html=True)


# -------------------- FOOTER --------------------

st.divider()

st.caption(
    "MILITARY TERRAIN ANALYSIS  •  Academic prototype  •  Local inference"
)