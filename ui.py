
import streamlit as st
import html


# ==================================================
# 1. WHITE AND BLUE THEME
# ==================================================

def apply_theme():
    st.markdown("""
    <style>
    .stApp {
        background: #F4F7FC;
        color: #253650;
    }

    [data-testid="stHeader"] {
        background: #F4F7FC;
    }

    .main .block-container {
        max-width: 1450px;
        padding: 1.8rem 2rem 2rem;
    }

    h1, h2, h3 {
        color: #173E76;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #E8F1FF, #F7FAFF);
        border-right: 1px solid #D6E4F6;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #174A8B;
    }

    /* Header banner */
    .hero-banner {
        background: linear-gradient(120deg, #FFFFFF, #E5F0FF);
        border: 1px solid #D4E4F8;
        border-radius: 18px;
        padding: 28px;
        margin-bottom: 12px;
        box-shadow: 0 5px 18px rgba(35, 77, 125, 0.06);
    }

    .hero-title {
        font-size: 32px;
        font-weight: 800;
        color: #174A8B;
        line-height: 1.3;
    }

    .hero-subtitle {
        font-size: 15px;
        line-height: 1.7;
        color: #5B708C;
        margin-top: 10px;
    }

    .hero-tag {
        display: inline-block;
        background: #DDEBFF;
        color: #1557A2;
        border: 1px solid #C8DDF8;
        border-radius: 20px;
        padding: 6px 12px;
        margin-top: 14px;
        font-size: 11px;
        font-weight: 750;
        letter-spacing: 0.6px;
    }

    /* Section headings */
    .section-heading {
        color: #173E76;
        font-size: 23px;
        font-weight: 750;
        margin-top: 18px;
        margin-bottom: 5px;
    }

    .section-description {
        color: #6A7D95;
        font-size: 14px;
        line-height: 1.6;
        margin-bottom: 18px;
    }

    /* Dashboard cards */
    .dashboard-card {
        background: #FFFFFF;
        border: 1px solid #DCE7F5;
        border-radius: 15px;
        padding: 19px;
        min-height: 110px;
        box-shadow: 0 4px 14px rgba(35, 73, 125, 0.045);
        overflow-wrap: anywhere;
    }

    .card-label {
        color: #70819A;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        margin-bottom: 10px;
    }

    .card-value {
        color: #1768C4;
        font-size: 25px;
        font-weight: 800;
        line-height: 1.35;
    }

    .card-description {
        color: #8492A6;
        font-size: 12px;
        margin-top: 7px;
        line-height: 1.5;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #DCE7F5;
        border-radius: 13px;
        padding: 15px;
    }

    [data-testid="stMetricValue"] {
        color: #1768C4;
    }

    /* Tabs */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #DCE6F3;
    }

    [data-testid="stTabs"] button[data-baseweb="tab"] {
        color: #526B89;
        font-weight: 650;
        padding: 12px 15px;
    }

    [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
        color: #1768C4;
        background: #E5F0FF;
        border-bottom: 3px solid #2878D0;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 9px;
        min-height: 42px;
        font-weight: 650;
    }

    div.stButton > button[kind="primary"] {
        background: #1768C4;
        color: white;
        border: 1px solid #1768C4;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #10549F;
        border-color: #10549F;
    }

    /* Input controls */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 9px;
        background: #FFFFFF;
    }

    /* Tables and expanders */
    [data-testid="stDataFrame"],
    [data-testid="stExpander"] {
        border: 1px solid #DCE7F5;
        border-radius: 11px;
        background: #FFFFFF;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        color: #8090A5;
        font-size: 12px;
        line-height: 1.8;
        padding: 22px 10px 8px;
        border-top: 1px solid #DCE6F3;
        margin-top: 30px;
    }

    /* Mobile layout */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem 0.8rem;
        }

        .hero-banner {
            padding: 20px;
        }

        .hero-title {
            font-size: 25px;
        }

        .section-heading {
            font-size: 20px;
        }

        .dashboard-card {
            padding: 14px;
        }

        .card-value {
            font-size: 21px;
        }
    }
    </style>
    """, unsafe_allow_html=True)


# ==================================================
# 2. APPLICATION HEADER
# ==================================================

def show_header():
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">
            🧠 Envisioned Speech Decoder
        </div>
        <div class="hero-subtitle">
            Explore EEG waveforms, inspect frequency representations,
            and generate imagined-speech class predictions using
            a saved EEGIS research dataset.
        </div>
        <div class="hero-tag">
            EEGIS · RESEARCH PROTOTYPE
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==================================================
# 3. DASHBOARD CARD
# ==================================================

def show_card(label, value, description=""):
    label = html.escape(str(label))
    value = html.escape(str(value))
    description = html.escape(str(description))

    description_html = ""

    if description:
        description_html = (
            f'<div class="card-description">{description}</div>'
        )

    st.markdown(
        f"""
        <div class="dashboard-card">
            <div class="card-label">{label}</div>
            <div class="card-value">{value}</div>
            {description_html}
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# 4. SECTION HEADING
# ==================================================

def show_section(title, description=""):
    title = html.escape(str(title))

    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True
    )

    if description:
        description = html.escape(str(description))

        st.markdown(
            f'<div class="section-description">'
            f'{description}</div>',
            unsafe_allow_html=True
        )


# ==================================================
# 5. SIDEBAR BRANDING
# ==================================================

def show_sidebar_brand():
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #FFFFFF, #DDEBFF);
        border: 1px solid #D0E1F6;
        border-radius: 14px;
        padding: 17px;
        margin-bottom: 12px;
    ">
        <div style="
            font-size: 24px;
            font-weight: 800;
            color: #174A8B;
        ">🧠 EEGIS</div>
        <div style="
            font-size: 12px;
            color: #647B98;
            margin-top: 5px;
        ">Envisioned Speech Decoder</div>
        <div style="
            display: inline-block;
            background: #E2EEFF;
            color: #1756A9;
            border-radius: 20px;
            padding: 5px 9px;
            margin-top: 10px;
            font-size: 10px;
            font-weight: 700;
        ">SAVED DATASET MODE</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Configure your trial and signal settings below.")
    st.divider()


# ==================================================
# 6. FOOTER
# ==================================================

def show_footer():
    st.markdown("""
    <div class="app-footer">
        <strong>Envisioned Speech Decoder</strong><br>
        EEGIS Research Prototype · Saved EEG trials only<br>
        Not a live thought-reading system or clinical device
    </div>
    """, unsafe_allow_html=True)
