"""
ClauseInsight — UI Theme
=========================

Centralized CSS theme for the entire ClauseInsight Streamlit app.
Provides a premium dark-mode design with glassmorphism, gradient
accents, micro-animations, and modern typography.

Usage:
    from src.ui.theme import apply_theme, page_header, feature_card, metric_card
    apply_theme()  # Call once at the top of every page

All styling is done via st.markdown(unsafe_allow_html=True) —
no external dependencies, no JavaScript, guaranteed Streamlit compat.
"""

import streamlit as st

# ── Google Fonts Import ────────────────────────────────────────────
GOOGLE_FONTS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
"""

# ── Master CSS ─────────────────────────────────────────────────────
MASTER_CSS = """
<style>
/* ═══════════════════════════════════════════════════════════════
   CLAUSEINSIGHT — PREMIUM DARK THEME
   ═══════════════════════════════════════════════════════════════ */

/* ── CSS Variables ─────────────────────────────────────────────── */
:root {
    --ci-bg-primary: #0B0F19;
    --ci-bg-secondary: #111827;
    --ci-bg-glass: transparent;
    --ci-border-glass: rgba(128, 128, 128, 0.2);
    --ci-border-hover: rgba(128, 128, 128, 0.5);
    --ci-text-primary: inherit;
    --ci-text-secondary: inherit;
    --ci-text-muted: inherit;
}

:root {
    --ci-accent-blue: #8B5CF6; /* Replaced blue with purple */
    --ci-accent-purple: #F59E0B; /* Replaced purple with gold */
    --ci-accent-teal: #D97706; /* Replaced teal with dark gold */
    --ci-accent-pink: #B45309;
    --ci-gradient-primary: linear-gradient(135deg, #4F46E5, #06B6D4, #0D9488);
    --ci-gradient-accent: linear-gradient(135deg, #4F46E5, #06B6D4);
    --ci-gradient-warm: linear-gradient(135deg, #06B6D4, #0D9488);
    --ci-shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.3);
    --ci-shadow-md: 0 4px 16px rgba(0, 0, 0, 0.4);
    --ci-shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.5);
    --ci-shadow-glow-blue: 0 0 20px rgba(139, 92, 246, 0.15);
    --ci-shadow-glow-purple: 0 0 20px rgba(245, 158, 11, 0.15);
    --ci-radius-sm: 8px;
    --ci-radius-md: 12px;
    --ci-radius-lg: 16px;
    --ci-radius-xl: 20px;
    --ci-transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Global Base ───────────────────────────────────────────────── */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Fade-in Animation ─────────────────────────────────────────── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

@keyframes shimmer {
    0% { background-position: -200% center; }
    100% { background-position: 200% center; }
}

@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-20px); }
    to { opacity: 1; transform: translateX(0); }
}

@keyframes float {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.8; }
}

@keyframes borderGlow {
    0%, 100% { border-color: rgba(139, 92, 246, 0.3); }
    50% { border-color: rgba(245, 158, 11, 0.5); }
}

/* ── Main Content Area ─────────────────────────────────────────── */
.main .block-container {
    animation: fadeIn 0.6s ease-out;
    max-width: 1200px;
}

/* ── Sidebar ───────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--ci-bg-primary) 0%, var(--ci-bg-secondary) 50%, var(--ci-bg-primary) 100%) !important;
    border-right: 1px solid var(--ci-border-glass) !important;
}

section[data-testid="stSidebar"] .stMarkdown {
    animation: slideInLeft 0.5s ease-out;
}

section[data-testid="stSidebar"] [data-testid="stSelectbox"] {
    animation: fadeIn 0.6s ease-out 0.2s both;
}

/* ── Headers ───────────────────────────────────────────────────── */
.stApp h1 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
    animation: fadeInUp 0.6s ease-out;
}

.stApp h2 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
    animation: fadeInUp 0.5s ease-out;
}

.stApp h3 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    animation: fadeInUp 0.5s ease-out;
}

/* ── Streamlit Native Buttons ────────────────────────────────────── */
.stApp button[kind="primary"],
.stApp button[data-testid="stBaseButton-primary"] {
    background: var(--ci-gradient-accent) !important;
    background-size: 200% 200% !important;
    border: none !important;
    border-radius: var(--ci-radius-md) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    padding: 0.5rem 1rem !important;
    color: #FFFFFF !important;
    transition: var(--ci-transition) !important;
    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.25) !important;
    animation: fadeInUp 0.6s ease-out;
}

.stApp button[kind="primary"]:hover,
.stApp button[data-testid="stBaseButton-primary"]:hover {
    opacity: 0.8 !important;
    box-shadow: 0 6px 25px rgba(139, 92, 246, 0.4) !important;
    animation: gradientShift 3s ease infinite !important;
    background-size: 200% 200% !important;
}

.stApp button[kind="primary"]:active,
.stApp button[data-testid="stBaseButton-primary"]:active {
    opacity: 0.7 !important;
}

.stApp button[kind="secondary"],
.stApp button[data-testid="stBaseButton-secondary"] {
    border: 1px solid var(--ci-border-glass) !important;
    border-radius: var(--ci-radius-md) !important;
    background: var(--ci-bg-glass) !important;
    backdrop-filter: blur(10px) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    transition: var(--ci-transition) !important;
}

.stApp button[kind="secondary"]:hover,
.stApp button[data-testid="stBaseButton-secondary"]:hover {
    border-color: var(--ci-accent-blue) !important;
    background: rgba(139, 92, 246, 0.08) !important;
    box-shadow: var(--ci-shadow-glow-blue) !important;
    opacity: 0.8 !important;
}

/* ── All other buttons (Select, Remove, Clear, etc.) ──────────── */
.stApp button {
    border-radius: var(--ci-radius-sm) !important;
    font-family: 'Inter', sans-serif !important;
    transition: var(--ci-transition) !important;
}

.stApp button:hover {
    opacity: 0.8 !important;
}

/* ── Metrics ───────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--ci-bg-glass) !important;
    border: 1px solid var(--ci-border-glass) !important;
    border-radius: var(--ci-radius-lg) !important;
    padding: 1.1rem 1.2rem !important;
    backdrop-filter: blur(12px) !important;
    transition: var(--ci-transition) !important;
    animation: fadeInUp 0.6s ease-out both;
    position: relative;
    overflow: hidden;
}

[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--ci-gradient-primary);
    background-size: 200% 200%;
    animation: gradientShift 4s ease infinite;
    border-radius: var(--ci-radius-lg) var(--ci-radius-lg) 0 0;
}

[data-testid="stMetric"]:hover {
    border-color: var(--ci-border-hover) !important;
    box-shadow: var(--ci-shadow-glow-blue) !important;
    opacity: 0.9 !important;
}

[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    color: var(--ci-text-secondary) !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 2rem !important;
}

/* stagger animation for metric columns */
[data-testid="stHorizontalBlock"] > div:nth-child(1) [data-testid="stMetric"] { animation-delay: 0.05s; }
[data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stMetric"] { animation-delay: 0.12s; }
[data-testid="stHorizontalBlock"] > div:nth-child(3) [data-testid="stMetric"] { animation-delay: 0.19s; }
[data-testid="stHorizontalBlock"] > div:nth-child(4) [data-testid="stMetric"] { animation-delay: 0.26s; }
[data-testid="stHorizontalBlock"] > div:nth-child(5) [data-testid="stMetric"] { animation-delay: 0.33s; }

/* ── Expanders ─────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--ci-bg-glass) !important;
    border: 1px solid var(--ci-border-glass) !important;
    border-radius: var(--ci-radius-md) !important;
    backdrop-filter: blur(10px) !important;
    margin-bottom: 0.6rem !important;
    transition: var(--ci-transition) !important;
    animation: fadeInUp 0.5s ease-out both;
    overflow: hidden;
}

[data-testid="stExpander"]:hover {
    border-color: var(--ci-border-hover) !important;
    box-shadow: var(--ci-shadow-sm) !important;
}

[data-testid="stExpander"] summary {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.8rem 1rem !important;
    transition: var(--ci-transition) !important;
}

[data-testid="stExpander"] summary:hover {
    color: var(--ci-accent-blue) !important;
}

/* ── Alerts / Info-boxes ───────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: var(--ci-radius-md) !important;
    backdrop-filter: blur(8px) !important;
    animation: fadeInUp 0.5s ease-out;
    font-family: 'Inter', sans-serif !important;
    border-left-width: 4px !important;
}

/* ── Status containers ─────────────────────────────────────────── */
[data-testid="stStatus"] {
    border-radius: var(--ci-radius-md) !important;
    border: 1px solid var(--ci-border-glass) !important;
    animation: fadeIn 0.6s ease-out;
}

/* ── File Uploader ─────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    animation: fadeInUp 0.6s ease-out;
}

[data-testid="stFileUploader"] section {
    border: 2px dashed var(--ci-border-glass) !important;
    border-radius: var(--ci-radius-lg) !important;
    background: var(--ci-bg-glass) !important;
    backdrop-filter: blur(10px) !important;
    transition: var(--ci-transition) !important;
    padding: 2rem !important;
}

[data-testid="stFileUploader"] section:hover {
    border-color: var(--ci-accent-blue) !important;
    background: rgba(245, 158, 11, 0.04) !important;
    box-shadow: var(--ci-shadow-glow-blue) !important;
}

/* ── Selectbox / Inputs ────────────────────────────────────────── */
[data-testid="stSelectbox"],
[data-testid="stMultiSelect"] {
    animation: fadeInUp 0.5s ease-out;
}

[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    border-radius: var(--ci-radius-sm) !important;
    border-color: var(--ci-border-glass) !important;
    transition: var(--ci-transition) !important;
}

[data-testid="stSelectbox"] > div > div:hover,
[data-testid="stMultiSelect"] > div > div:hover {
    border-color: var(--ci-accent-blue) !important;
}

/* ── Chat Messages ─────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: var(--ci-bg-glass) !important;
    border: 1px solid var(--ci-border-glass) !important;
    border-radius: var(--ci-radius-lg) !important;
    backdrop-filter: blur(12px) !important;
    padding: 1rem 1.2rem !important;
    margin-bottom: 0.8rem !important;
    animation: fadeInUp 0.4s ease-out !important;
    transition: var(--ci-transition) !important;
}

[data-testid="stChatMessage"]:hover {
    border-color: var(--ci-border-hover) !important;
    box-shadow: var(--ci-shadow-sm) !important;
}

/* User messages — accent border */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-left: 3px solid var(--ci-accent-blue) !important;
}

/* Assistant messages — purple accent */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 3px solid var(--ci-accent-purple) !important;
}

/* ── Chat Input ────────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    border-radius: var(--ci-radius-lg) !important;
    animation: fadeInUp 0.6s ease-out 0.3s both;
}

[data-testid="stChatInput"] textarea {
    font-family: 'Inter', sans-serif !important;
    border-radius: var(--ci-radius-md) !important;
}

/* ── Dividers ──────────────────────────────────────────────────── */
[data-testid="stApp"] hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, var(--ci-border-glass), var(--ci-accent-blue), var(--ci-border-glass), transparent) !important;
    margin: 1.5rem 0 !important;
    animation: fadeIn 0.8s ease-out;
}

/* ── Spinner ───────────────────────────────────────────────────── */
[data-testid="stSpinner"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── Slider ────────────────────────────────────────────────────── */
[data-testid="stSlider"] {
    animation: fadeInUp 0.5s ease-out;
}

/* ── Toggle ────────────────────────────────────────────────────── */
[data-testid="stToggle"] {
    animation: fadeInUp 0.5s ease-out;
}

/* ── Download button ───────────────────────────────────────────── */
[data-testid="stDownloadButton"] button {
    background: var(--ci-bg-glass) !important;
    border: 1px solid var(--ci-border-glass) !important;
    border-radius: var(--ci-radius-md) !important;
    backdrop-filter: blur(10px) !important;
    transition: var(--ci-transition) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
}

[data-testid="stDownloadButton"] button:hover {
    border-color: var(--ci-accent-teal) !important;
    background: rgba(251, 191, 36, 0.08) !important;
    box-shadow: 0 0 20px rgba(251, 191, 36, 0.15) !important;
    opacity: 0.8 !important;
}

/* Stop Streamlit buttons from bouncing/moving on hover or click */
.stButton button {
    transition: background-color 0.2s ease, color 0.2s ease, border-color 0.2s ease, opacity 0.2s ease !important;
    transform: none !important;
}
.stButton button:hover, .stButton button:active, .stButton button:focus {
    transform: none !important;
    opacity: 0.85;
}

/* ── Custom Scrollbar ──────────────────────────────────────────── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.2);
}

/* ── Tabs ──────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: var(--ci-radius-sm) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    transition: var(--ci-transition) !important;
}

.stTabs [data-baseweb="tab"]:hover {
    background: var(--ci-bg-glass) !important;
}

/* ── Columns equal height ──────────────────────────────────────── */
[data-testid="stHorizontalBlock"] {
    align-items: stretch;
}

</style>
"""

LIGHT_THEME_CSS = """
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stBottom"], [data-testid="stBottomBlockContainer"] {
    background-color: #FFFFFF !important;
    color: #31333F !important;
}
.stApp p, .stApp span, .stApp div, .stApp h1, .stApp h2, .stApp h3, .stApp label, .stApp li {
    color: #31333F;
}
section[data-testid="stSidebar"] {
    background-color: #0E1117 !important;
}
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] a, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] li {
    color: #FAFAFA !important;
}
.stApp div[style*="-webkit-text-fill-color:transparent"] {
    color: transparent !important;
}
</style>
"""

DARK_THEME_CSS = """
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stBottom"], [data-testid="stBottomBlockContainer"] {
    background-color: #0E1117 !important;
    color: #FAFAFA !important;
}
.stApp p, .stApp span, .stApp div, .stApp h1, .stApp h2, .stApp h3, .stApp label, .stApp li {
    color: #FAFAFA;
}
section[data-testid="stSidebar"] {
    background-color: #0E1117 !important;
}
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] a, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] li {
    color: #FAFAFA !important;
}
.stApp div[style*="-webkit-text-fill-color:transparent"] {
    color: transparent !important;
}
</style>
"""

def apply_theme():
    """Inject the master theme CSS. Call once at the top of every page."""
    st.markdown(GOOGLE_FONTS, unsafe_allow_html=True)
    st.markdown(MASTER_CSS, unsafe_allow_html=True)
    theme_mode = st.session_state.get("theme_mode", "dark")
    if theme_mode == "light":
        st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)
    else:
        st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)


def gradient_header(title: str, subtitle: str = "", emoji: str = ""):
    """Render a large animated gradient text header."""
    emoji_html = f'<span style="font-size:2.2rem;margin-right:0.5rem;">{emoji}</span>' if emoji else ""
    subtitle_html = (
        f'<p style="color:var(--ci-text-secondary);font-size:1.15rem;font-weight:400;'
        f'margin-top:0.5rem;font-family:Inter,sans-serif;animation:fadeInUp 0.7s ease-out 0.2s both;">'
        f'{subtitle}</p>'
    ) if subtitle else ""

    st.markdown(f"""
<div style="animation:fadeInUp 0.6s ease-out;margin-bottom:1.5rem;">
    <h1 style="
        font-family:'Inter',sans-serif;
        font-weight:800;
        font-size:2.5rem;
        letter-spacing:-0.03em;
        margin:0;
        display:flex;
        align-items:center;
    ">
        {emoji_html}
        <span style="
        background:linear-gradient(135deg, #4F46E5, #06B6D4, #0D9488, #4F46E5);
            background-size:300% 300%;
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
            background-clip:text;
            animation:gradientShift 6s ease infinite;
        ">{title}</span>
    </h1>
    {subtitle_html}
</div>
""", unsafe_allow_html=True)


def glass_card(content: str, border_color: str = "rgba(255,255,255,0.08)", extra_style: str = ""):
    """Render content inside a glassmorphism card."""
    st.markdown(f"""
<div style="border:1px solid {border_color}; border-radius:16px; padding:1.5rem; transition:all 0.3s cubic-bezier(0.4,0,0.2,1); animation:fadeInUp 0.6s ease-out both; {extra_style}">
{content}
</div>
""", unsafe_allow_html=True)


def feature_card(icon: str, title: str, description: str, delay: str = "0s", gradient: str = "135deg, #4F46E5, #06B6D4"):
    """Render a feature card with icon, title, and description for the landing page."""
    return f"""
<div style="border:1px solid var(--ci-border-glass); border-radius:20px; padding:2rem 1.5rem; transition:all 0.3s cubic-bezier(0.4,0,0.2,1); animation:fadeInUp 0.6s ease-out {delay} both; cursor:default; height:100%; position:relative; overflow:hidden;" onmouseover="this.style.borderColor='var(--ci-border-hover)'; this.style.boxShadow='var(--ci-shadow-md)';" onmouseout="this.style.borderColor='var(--ci-border-glass)'; this.style.boxShadow='none';">
    <div style="position:absolute;top:0;left:0;right:0;height:3px; background:linear-gradient({gradient}); border-radius:20px 20px 0 0;"></div>
    <div style="font-size:2.8rem; margin-bottom:1rem; animation:float 3s ease-in-out infinite; animation-delay:{delay};">{icon}</div>
    <h3 style="font-family:'Inter',sans-serif; font-weight:700; font-size:1.2rem; margin:0 0 0.8rem 0; color:var(--ci-text-primary); letter-spacing:-0.01em;">{title}</h3>
    <p style="font-family:'Inter',sans-serif; color:var(--ci-text-secondary); font-size:0.92rem; line-height:1.6; margin:0;">{description}</p>
</div>
"""


def stat_badge(label: str, value: str, color: str = "#8B5CF6"):
    """Render an inline stat badge."""
    return f"""
<div style="
    display:inline-flex;
    align-items:center;
    gap:0.5rem;
    background:var(--ci-bg-glass);
    border:1px solid var(--ci-border-glass);
    border-radius:10px;
    padding:0.4rem 0.9rem;
    font-family:'Inter',sans-serif;
    animation:fadeInUp 0.5s ease-out both;
">
    <span style="color:{color};font-weight:700;font-size:1.1rem;">{value}</span>
    <span style="color:var(--ci-text-secondary);font-size:0.82rem;font-weight:500;">{label}</span>
</div>
"""


def risk_badge_html(level: str, size: str = "normal") -> str:
    """Return HTML for a styled risk badge with appropriate color and glow."""
    colors = {
        "HIGH": ("#FF4B4B", "rgba(255,75,75,0.2)", "pulse 2s ease-in-out infinite"),
        "MEDIUM": ("#F59E0B", "rgba(245,158,11,0.15)", "none"),
        "LOW": ("#10B981", "rgba(16,185,129,0.15)", "none"),
        "UNKNOWN": ("#6B7280", "rgba(107,114,128,0.15)", "none"),
    }
    color, glow_color, animation = colors.get(level, colors["UNKNOWN"])
    font_size = "0.85rem" if size == "normal" else "0.75rem"
    padding = "4px 12px" if size == "normal" else "2px 8px"

    return (
        f'<span style="'
        f'display:inline-block;'
        f'background:{glow_color};'
        f'color:{color};'
        f'border:1px solid {color};'
        f'padding:{padding};'
        f'border-radius:8px;'
        f'font-family:Inter,sans-serif;'
        f'font-weight:700;'
        f'font-size:{font_size};'
        f'letter-spacing:0.05em;'
        f'animation:{animation};'
        f'box-shadow:0 0 12px {glow_color};'
        f'">{level}</span>'
    )


def obligation_badge_html(ob_type: str, color: str) -> str:
    """Return HTML for a styled obligation type badge."""
    return (
        f'<span style="'
        f'display:inline-block;'
        f'background:rgba({_hex_to_rgb(color)},0.15);'
        f'color:{color};'
        f'border:1px solid {color};'
        f'padding:4px 12px;'
        f'border-radius:8px;'
        f'font-family:Inter,sans-serif;'
        f'font-weight:700;'
        f'font-size:0.85rem;'
        f'letter-spacing:0.03em;'
        f'">{ob_type}</span>'
    )


def section_header(title: str, subtitle: str = ""):
    """Render a styled section header with optional subtitle."""
    sub_html = (
        f'<p style="color:var(--ci-text-secondary);font-size:0.9rem;margin:0.3rem 0 0 0;'
        f'font-family:Inter,sans-serif;">{subtitle}</p>'
    ) if subtitle else ""

    st.markdown(f"""
<div style="margin:1.2rem 0 1rem 0;animation:fadeInUp 0.5s ease-out;">
    <h2 style="
        font-family:'Inter',sans-serif;
        font-weight:700;
        font-size:1.4rem;
        color:var(--ci-text-primary);
        margin:0;
        letter-spacing:-0.01em;
    ">{title}</h2>
    {sub_html}
</div>
""", unsafe_allow_html=True)




def top_bar():
    """Render a top bar with a theme toggle."""
    col1, col2 = st.columns([8, 1])
    with col2:
        theme_mode = st.session_state.get("theme_mode", "dark")
        icon = "☀️ Light" if theme_mode == "dark" else "🌙 Dark"
        if st.button(icon, key="theme_toggle"):
            st.session_state["theme_mode"] = "light" if theme_mode == "dark" else "dark"
            st.rerun()

def sidebar_brand():
    """Render a branded sidebar header."""
    st.sidebar.markdown("""
<div style="
    text-align:center;
    padding:0.8rem 0 1.2rem 0;
    border-bottom:1px solid var(--ci-border-glass);
    margin-bottom:1rem;
    animation:fadeIn 0.8s ease-out;
">
    <div style="font-size:2rem;margin-bottom:0.3rem;">⚖️</div>
    <div style="
        font-family:'Inter',sans-serif;
        font-weight:700;
        font-size:1.1rem;
        background:linear-gradient(135deg, #4F46E5, #06B6D4);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
        background-clip:text;
        letter-spacing:-0.01em;
    ">ClauseInsight</div>
    <div style="
        color:var(--ci-text-secondary);
        font-size:0.72rem;
        font-family:'Inter',sans-serif;
        font-weight:500;
        letter-spacing:0.05em;
        text-transform:uppercase;
        margin-top:0.2rem;
    ">Contract Intelligence</div>
</div>
""", unsafe_allow_html=True)


def footer():
    """Render a subtle footer."""
    st.markdown("""
<div style="
    text-align:center;
    padding:2rem 0 1rem 0;
    margin-top:2rem;
    border-top:1px solid var(--ci-border-glass);
    animation:fadeIn 1s ease-out;
">
    <p style="
        color:var(--ci-text-muted);
        font-size:0.78rem;
        font-family:'Inter',sans-serif;
        font-weight:400;
        margin:0;
    ">
        Built by <strong style="color:var(--ci-text-secondary);">Tejas T. P.</strong> ·
        Foundations of Applied ML Internship 2026
    </p>
</div>
""", unsafe_allow_html=True)


def sidebar_footer():
    """Render the bottom elements of the sidebar, like the Exit button."""
    with st.sidebar:
        # Push to the bottom using empty space
        st.markdown("<div style='flex-grow: 1;'></div>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Inject CSS and JS to style the Exit button red and handle hover natively
        st.markdown("""
        <style>
        .exit-btn-custom {
            color: #ef4444 !important;
            border-color: rgba(239, 68, 68, 0.3) !important;
            transition: all 0.2s ease !important;
        }
        .exit-btn-custom p {
            color: #ef4444 !important;
        }
        .exit-btn-custom .material-symbols-rounded {
            color: #ef4444 !important;
            transition: all 0.2s ease !important;
        }
        
        .exit-btn-custom:hover {
            background-color: #ef4444 !important;
            border-color: #ef4444 !important;
        }
        .exit-btn-custom:hover p {
            color: #ffffff !important;
        }
        .exit-btn-custom:hover .material-symbols-rounded {
            color: #ffffff !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        import streamlit.components.v1 as components
        components.html(
            """
            <script>
            const doc = window.parent.document;
            function styleExitBtn() {
                const buttons = doc.querySelectorAll('button');
                buttons.forEach(b => {
                    if (b.textContent.includes('Exit')) {
                        b.classList.add('exit-btn-custom');
                    }
                });
            }
            styleExitBtn();
            // run occasionally in case of re-renders
            setInterval(styleExitBtn, 1000);
            </script>
            """,
            height=0, width=0
        )
        
        if st.button("Exit", icon=":material/logout:", use_container_width=True, help="Stop the Streamlit server"):
            # Inject beautiful shutdown UI
            components.html(
                """
                <script>
                const doc = window.parent.document;
                doc.body.innerHTML = `
                    <div style="
                        display: flex; 
                        flex-direction: column; 
                        align-items: center; 
                        justify-content: center; 
                        height: 100vh; 
                        width: 100vw;
                        background-color: #0E1117; 
                        background-image: radial-gradient(circle at 50% 50%, rgba(79, 70, 229, 0.15) 0%, #0E1117 60%);
                        color: #FAFAFA; 
                        font-family: 'Inter', sans-serif;
                        margin: 0;
                        overflow: hidden;
                    ">
                        <div style="animation: fadeInUp 0.8s ease-out; text-align: center;">
                            <div style="font-size: 5rem; margin-bottom: 1rem; animation: float 3s ease-in-out infinite;">⚖️</div>
                            <h1 style="
                                font-size: 4.5rem; 
                                font-weight: 800;
                                background: linear-gradient(135deg, #4F46E5, #06B6D4); 
                                -webkit-background-clip: text; 
                                -webkit-text-fill-color: transparent;
                                margin: 0 0 0.5rem 0;
                                letter-spacing: -0.02em;
                            ">ClauseInsight</h1>
                            <h2 style="font-size: 1.8rem; color: #94A3B8; font-weight: 400; margin: 0;">Thanks for using our platform.</h2>
                            
                            <div style="margin-top: 3rem; display: flex; flex-direction: column; align-items: center; gap: 1rem;">
                                <div id="shutdown-loader" style="
                                    width: 40px; 
                                    height: 40px; 
                                    border: 3px solid rgba(255,255,255,0.1); 
                                    border-top-color: #06B6D4; 
                                    border-radius: 50%; 
                                    animation: spin 1s linear infinite;
                                "></div>
                                <p style="color: #64748B; font-size: 0.95rem;">Shutting down server... You may safely close this tab.</p>
                            </div>
                            
                            <p style="
                                font-family: 'Fredoka One', cursive;
                                font-weight: 400;
                                font-size: 15px;
                                color: #9CA3AF;
                                letter-spacing: 0.05em;
                                text-align: center;
                                margin-top: 32px;
                            ">Developed by Tejas T. P.</p>
                        </div>
                        <style>
                            @import url('https://fonts.googleapis.com/css2?family=Fredoka+One&display=swap');
                            @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
                            @keyframes float { 0% { transform: translateY(0px); } 50% { transform: translateY(-15px); } 100% { transform: translateY(0px); } }
                            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
                            @keyframes scaleIn { from { transform: scale(0); } to { transform: scale(1); } }
                        </style>
                    </div>
                `;
                
                const script = doc.createElement('script');
                script.textContent = `
                    setTimeout(() => {
                        const loader = document.getElementById('shutdown-loader');
                        if (loader) {
                            loader.style.animation = 'none';
                            loader.style.border = 'none';
                            loader.style.display = 'flex';
                            loader.style.alignItems = 'center';
                            loader.style.justifyContent = 'center';
                            loader.style.backgroundColor = '#10B981';
                            loader.style.borderRadius = '50%';
                            loader.innerHTML = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" style="animation: scaleIn 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) both;"><path d="M20 6L9 17l-5-5"/></svg>';
                        }
                    }, 3000);
                `;
                doc.body.appendChild(script);
                </script>
                """,
                height=0, width=0
            )
            
            # Start background thread to kill the server after the JS has time to render
            import threading
            import time
            import os
            import signal
            
            def delayed_kill():
                time.sleep(2)  # Give the frontend 2 seconds to receive the JS and update the DOM
                os.kill(os.getpid(), signal.SIGINT)
                
            threading.Thread(target=delayed_kill, daemon=True).start()


def _hex_to_rgb(hex_color: str) -> str:
    """Convert #RRGGBB to 'R,G,B' string for use in rgba()."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"{r},{g},{b}"
