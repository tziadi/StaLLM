"""Visual system for the StaLLM workbench.

One stylesheet, a handful of HTML components and an Altair theme. The look is
deliberately sober (paper-like surfaces, serif headings, one accent colour) so
that screenshots can go straight into a paper or a talk.
"""
from __future__ import annotations

from html import escape
from typing import Iterable

import altair as alt
import streamlit as st

BRAND_NAME = "StaLLM"
BRAND_TAGLINE = "A benchmark with built-in controls for LLMs in software maintenance"

# Chart palette: first slots of the validated reference palette (dataviz skill),
# plus a neutral for baselines / chance so that the model always carries colour.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BASELINE = "#8a8f98"
CHANCE_BAND = "#d9dbe0"
INK = "#1b1f27"
INK_MUTED = "#5b6270"
GRID = "#e7e8ea"

SERIF = "'Source Serif 4', 'Source Serif Pro', Georgia, 'Times New Roman', serif"
SANS = "Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600;8..60,700&display=swap');

:root{
  --bg:#f7f7f5; --surface:#ffffff; --surface-2:#f2f3f1;
  --ink:#1b1f27; --ink-2:#3a404c; --muted:#5b6270; --faint:#8a8f98;
  --line:#e3e4e0; --line-2:#d2d4cf;
  --accent:#1f3a5f; --accent-2:#2a78d6; --accent-soft:#e8eef6;
  --good:#24724f; --good-soft:#e5f2eb;
  --warn:#8a5a00; --warn-soft:#fbf1dc;
  --bad:#a23b2a; --bad-soft:#f8e7e3;
  --hl:rgba(237,161,0,.22);
  --serif:__SERIF__; --sans:__SANS__; --mono:__MONO__;
}

/* ---------- shell ---------- */
html, body, .stApp{ background:var(--bg) !important; color:var(--ink); }
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea,
.stApp button, .stApp [data-baseweb], .stApp [data-testid="stMarkdownContainer"]{
  font-family:var(--sans);
}
#MainMenu, footer, .stDeployButton, [data-testid="stDecoration"]{ display:none !important; }
header[data-testid="stHeader"]{ background:transparent !important; }
.block-container{ padding-top:2.2rem !important; padding-bottom:4rem !important; max-width:1320px !important; }

.stApp h1, .stApp h2, .stApp h3{
  font-family:var(--serif) !important; color:var(--ink); font-weight:600; letter-spacing:-.005em;
}
.stApp h1{ font-size:2.05rem; line-height:1.15; }
.stApp h2{ font-size:1.45rem; margin-top:1.6rem; }
.stApp h3{ font-size:1.15rem; }
.stApp h4{ font-family:var(--sans) !important; font-size:.95rem; font-weight:600; color:var(--ink-2); }
.stMarkdown p, .stMarkdown li{ line-height:1.6; color:var(--ink-2); }
[data-testid="stCaptionContainer"], .stCaption{ color:var(--muted) !important; }
.stApp code{ font-family:var(--mono) !important; font-size:.86em; background:var(--surface-2); color:var(--ink); padding:.1em .35em; border-radius:4px; }
.stApp pre, .stCode, .stCode *{ font-family:var(--mono) !important; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"]{ background:var(--surface) !important; border-right:1px solid var(--line) !important; }
section[data-testid="stSidebar"] > div:first-child{ padding-top:1rem; }
[data-testid="stSidebarNav"] a span{ font-family:var(--sans); font-size:.9rem; color:var(--ink-2); }
[data-testid="stSidebarNav"] a[aria-current="page"]{ background:var(--accent-soft) !important; }
[data-testid="stSidebarNav"] a[aria-current="page"] span{ color:var(--accent) !important; font-weight:600; }
[data-testid="stSidebarNavSeparator"]{ border-color:var(--line) !important; }
[data-testid="stSidebarNav"] header, [data-testid="stNavSectionHeader"]{
  font-family:var(--sans) !important; font-size:.7rem !important; font-weight:600 !important;
  letter-spacing:.09em; text-transform:uppercase; color:var(--faint) !important;
}
.sb-brand{ display:flex; gap:.6rem; align-items:center; padding:.2rem .1rem .9rem; border-bottom:1px solid var(--line); margin-bottom:.4rem; }
.sb-brand img{ width:30px; height:30px; border-radius:7px; }
.sb-brand .name{ font-family:var(--serif); font-weight:700; font-size:1.25rem; color:var(--ink); line-height:1; }
.sb-brand .sub{ font-size:.72rem; color:var(--muted); margin-top:.2rem; }
.sb-meta{ font-size:.74rem; color:var(--muted); line-height:1.55; padding:.6rem .1rem 0; border-top:1px solid var(--line); margin-top:.8rem; }
.sb-meta code{ font-size:.72rem !important; }

/* ---------- page header ---------- */
.ph{ padding:0 0 1.1rem; margin-bottom:1.4rem; border-bottom:1px solid var(--line); }
.ph .eyebrow{ font-size:.72rem; font-weight:600; letter-spacing:.11em; text-transform:uppercase; color:var(--accent-2); margin-bottom:.45rem; }
.ph h1{ margin:0 !important; padding:0 !important; }
.ph .lede{ margin:.55rem 0 0; color:var(--muted); font-size:1.02rem; line-height:1.55; max-width:70ch; }

/* ---------- cards & stats ---------- */
.card{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:1.05rem 1.15rem; height:100%; }
.card h4{ margin:0 0 .35rem !important; }
.card p{ margin:.2rem 0 0; color:var(--muted); font-size:.9rem; line-height:1.55; }
.card .kicker{ font-size:.7rem; letter-spacing:.09em; text-transform:uppercase; color:var(--faint); font-weight:600; }
.stat{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:.85rem 1rem; }
.stat .label{ font-size:.76rem; color:var(--muted); font-weight:500; }
.stat .value{ font-family:var(--serif); font-size:1.75rem; font-weight:600; color:var(--ink); line-height:1.2; font-variant-numeric:tabular-nums; }
.stat .note{ font-size:.75rem; color:var(--faint); margin-top:.1rem; }
.chip{ display:inline-flex; align-items:center; gap:.35rem; padding:.16rem .55rem; border-radius:999px; font-size:.74rem; font-weight:600; border:1px solid transparent; white-space:nowrap; }
.chip-good{ background:var(--good-soft); color:var(--good); border-color:#cfe6d9; }
.chip-warn{ background:var(--warn-soft); color:var(--warn); border-color:#f0dfb8; }
.chip-bad{ background:var(--bad-soft); color:var(--bad); border-color:#efd0c9; }
.chip-neutral{ background:var(--surface-2); color:var(--ink-2); border-color:var(--line); }
.chip-accent{ background:var(--accent-soft); color:var(--accent); border-color:#d3dfee; }
.control{ background:var(--surface); border:1px solid var(--line); border-left:3px solid var(--accent); border-radius:8px; padding:1rem 1.15rem; margin-bottom:.9rem; }
.control .head{ display:flex; justify-content:space-between; align-items:baseline; gap:1rem; flex-wrap:wrap; }
.control .title{ font-family:var(--serif); font-size:1.12rem; font-weight:600; color:var(--ink); }
.control .threat{ font-size:.85rem; color:var(--muted); margin:.3rem 0 .55rem; }
.control .finding{ font-size:.92rem; color:var(--ink-2); line-height:1.55; }
.control .src{ font-size:.74rem; color:var(--faint); margin-top:.5rem; font-family:var(--mono); }
.note-box{ background:var(--surface-2); border:1px solid var(--line); border-radius:8px; padding:.75rem .95rem; font-size:.86rem; color:var(--ink-2); line-height:1.55; }
.muted{ color:var(--muted); }
.footnote{ color:var(--faint); font-size:.8rem; margin-top:.5rem; }

/* ---------- widgets ---------- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button{
  border-radius:7px !important; font-weight:600 !important; font-family:var(--sans) !important;
  border:1px solid var(--line-2) !important; background:var(--surface) !important; color:var(--ink) !important;
  box-shadow:none !important; padding:.42rem .95rem !important;
}
.stButton > button:hover, .stDownloadButton > button:hover{ border-color:var(--accent) !important; color:var(--accent) !important; }
.stButton > button[kind="primary"], [data-testid="baseButton-primary"]{
  background:var(--accent) !important; border-color:var(--accent) !important; color:#fff !important;
}
.stButton > button[kind="primary"] p, [data-testid="baseButton-primary"] p{ color:#fff !important; }
.stButton > button[kind="primary"]:hover, [data-testid="baseButton-primary"]:hover{ background:#162b47 !important; color:#fff !important; }
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-testid="stNumberInput"] input,
[data-baseweb="select"] > div{
  border-radius:7px !important; border-color:var(--line-2) !important; background:var(--surface) !important;
}
[data-testid="stTextArea"] textarea{ font-family:var(--mono) !important; font-size:12.5px !important; }
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label{ font-weight:600 !important; color:var(--ink-2) !important; font-size:.84rem !important; }
[data-testid="stFileUploader"] section{ border:1px dashed var(--line-2) !important; background:var(--surface) !important; border-radius:8px !important; }
[data-testid="stTabs"] [role="tablist"]{ gap:.2rem; border-bottom:1px solid var(--line); }
[data-testid="stTabs"] [role="tab"]{ padding:.55rem .9rem; }
[data-testid="stTabs"] [role="tab"] p{ font-weight:600; color:var(--muted); font-size:.9rem; }
[data-testid="stTabs"] [aria-selected="true"] p{ color:var(--accent) !important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{ background:var(--accent) !important; }
[data-testid="stExpander"] details{ border:1px solid var(--line) !important; border-radius:8px !important; background:var(--surface) !important; }
[data-testid="stExpander"] summary p{ font-weight:600; color:var(--ink-2); }
[data-testid="stAlert"]{ border-radius:8px !important; }
[data-testid="stDataFrame"]{ border:1px solid var(--line) !important; border-radius:8px !important; overflow:hidden; }
[data-testid="stMetric"]{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:.75rem .95rem; }
[data-testid="stMetricValue"]{ font-family:var(--serif) !important; }
[data-testid="stVegaLiteChart"]{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:.6rem .4rem .2rem; }

/* ---------- legacy components kept by the experiment pages ---------- */
.section{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:1rem 1.1rem; margin:.6rem 0 1rem; }
.section h3{ margin-top:0 !important; }
.right-settings-title{ font-size:.7rem; font-weight:600; letter-spacing:.09em; text-transform:uppercase; color:var(--faint); padding:.1rem 0 .5rem; border-bottom:1px solid var(--line); margin-bottom:.6rem; }
.badges{ margin-top:.4rem; }
.badge, .pill{ display:inline-flex; align-items:center; gap:.35rem; padding:.2rem .6rem; border:1px solid var(--line); border-radius:999px; font-size:.76rem; font-weight:500; background:var(--surface-2); color:var(--ink-2); margin:.15rem .3rem .35rem 0; }
.badge b, .pill b{ color:var(--ink); font-weight:600; }
.kpi{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:.8rem 1rem; min-height:72px; display:flex; flex-direction:column; justify-content:center; }
.kpi-label{ color:var(--muted); font-size:.78rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.kpi-value{ font-family:var(--serif); font-size:1.55rem; font-weight:600; color:var(--ink); font-variant-numeric:tabular-nums; }
.kpi-ok{ border-left:3px solid var(--good); } .kpi-warn{ border-left:3px solid #c98500; } .kpi-bad{ border-left:3px solid var(--bad); }

.codebox{ font-family:var(--mono); font-size:12.5px; border:1px solid var(--line); border-radius:8px; overflow:hidden; background:var(--surface); }
.codebox table{ border-collapse:collapse; width:100%; }
.codebox td{ vertical-align:top; padding:0; }
.codebox .gutter{ width:3.2em; background:var(--surface-2); color:var(--faint); text-align:right; user-select:none; }
.codebox .code{ white-space:pre; }
.codebox .line{ padding:1px 10px; }
.codebox .ln{ padding:1px 8px; }
.codebox .hl{ background:var(--hl); }
.codebox .gt{ background:#fdf0e6; box-shadow:inset 3px 0 #eb6834; }
.codebox .llm{ background:#eaf2fc; box-shadow:inset 3px 0 #2a78d6; }
.codebox .both{ background:#e6f5ee; box-shadow:inset 3px 0 #1baf7a; }
.code-legend{ display:flex; gap:8px; margin:.4rem 0 .6rem; flex-wrap:wrap; }
.legend-chip{ display:inline-flex; align-items:center; border-radius:999px; padding:3px 9px; font-size:12px; font-weight:500; border:1px solid var(--line); background:var(--surface); color:var(--ink-2); }
.legend-dot{ width:8px; height:8px; border-radius:999px; margin-right:6px; }
.dot-gt{ background:#eb6834; } .dot-llm{ background:#2a78d6; } .dot-both{ background:#1baf7a; }

.pro-table-wrap{ border:1px solid var(--line); border-radius:8px; overflow:auto; background:var(--surface); margin:.5rem 0 1rem; }
.pro-table{ width:100%; border-collapse:collapse; font-size:13px; line-height:1.4; color:var(--ink); }
.pro-table th{ position:sticky; top:0; z-index:1; background:var(--surface-2); color:var(--muted); text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.06em; font-weight:600; padding:9px 12px; border-bottom:1px solid var(--line); white-space:nowrap; }
.pro-table td{ padding:8px 12px; border-bottom:1px solid #eeefec; vertical-align:middle; }
.pro-table tr:last-child td{ border-bottom:none; }
.pro-table tr:hover td{ background:#fafaf8; }
.pro-table td.num{ text-align:right; font-variant-numeric:tabular-nums; }
.pro-table td.best{ font-weight:700; }
.pro-table.wrap td{ white-space:normal; vertical-align:top; }
.pro-table.wrap td:first-child{ font-weight:600; white-space:nowrap; }
.pro-table .file-cell{ font-family:var(--mono); font-size:12px; max-width:620px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.verdict-badge{ display:inline-flex; align-items:center; border-radius:999px; padding:2px 9px; font-size:12px; font-weight:600; border:1px solid transparent; white-space:nowrap; }
.verdict-matched{ color:var(--good); background:var(--good-soft); border-color:#cfe6d9; }
.verdict-partial, .verdict-mismatch{ color:var(--warn); background:var(--warn-soft); border-color:#f0dfb8; }
.verdict-missed{ color:var(--bad); background:var(--bad-soft); border-color:#efd0c9; }
.verdict-extra{ color:#4b3fa8; background:#eeecfa; border-color:#dcd8f4; }
.verdict-true_negative{ color:var(--ink-2); background:var(--surface-2); border-color:var(--line); }
.issue-pill{ display:inline-flex; border-radius:6px; padding:2px 7px; background:var(--surface-2); color:var(--ink-2); font-weight:500; white-space:nowrap; }
.metric-bar{ min-width:92px; }
.metric-bar-track{ height:6px; border-radius:999px; background:#ebece8; overflow:hidden; }
.metric-bar-fill{ height:100%; border-radius:999px; background:var(--accent-2); }
.metric-bar-label{ margin-top:3px; color:var(--muted); font-size:11px; font-variant-numeric:tabular-nums; text-align:right; }
</style>
"""


def inject_theme() -> None:
    css = _CSS.replace("__SERIF__", SERIF).replace("__SANS__", SANS).replace("__MONO__", MONO)
    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------------- components
def page_header(title: str, lede: str = "", eyebrow: str = "") -> None:
    eb = f'<div class="eyebrow">{escape(eyebrow)}</div>' if eyebrow else ""
    ld = f'<p class="lede">{lede}</p>' if lede else ""
    st.markdown(f'<div class="ph">{eb}<h1>{escape(title)}</h1>{ld}</div>', unsafe_allow_html=True)


def chip(text: str, tone: str = "neutral") -> str:
    return f'<span class="chip chip-{tone}">{escape(text)}</span>'


def stat_html(label: str, value: str, note: str = "") -> str:
    nt = f'<div class="note">{escape(note)}</div>' if note else ""
    return f'<div class="stat"><div class="label">{escape(label)}</div><div class="value">{escape(value)}</div>{nt}</div>'


def stat_row(items: Iterable[tuple[str, str, str]]) -> None:
    items = list(items)
    cols = st.columns(len(items))
    for col, (label, value, note) in zip(cols, items):
        col.markdown(stat_html(label, value, note), unsafe_allow_html=True)


def card(title: str, body: str, kicker: str = "") -> str:
    k = f'<div class="kicker">{escape(kicker)}</div>' if kicker else ""
    return f'<div class="card">{k}<h4>{escape(title)}</h4><p>{body}</p></div>'


def note(body: str) -> None:
    st.markdown(f'<div class="note-box">{body}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- charts
def _altair_theme() -> dict:
    return {
        "config": {
            "background": "#ffffff",
            "font": SANS,
            "view": {"stroke": None},
            "padding": {"top": 8, "right": 12, "bottom": 4, "left": 4},
            "title": {"font": SANS, "fontSize": 13, "fontWeight": 600, "color": INK, "anchor": "start", "offset": 10},
            "axis": {
                "labelFont": SANS, "titleFont": SANS, "labelFontSize": 11, "titleFontSize": 11.5,
                "labelColor": INK_MUTED, "titleColor": INK_MUTED, "titleFontWeight": 500,
                "gridColor": GRID, "domainColor": "#c9cbcf", "tickColor": "#c9cbcf",
            },
            "axisY": {"titlePadding": 8},
            "legend": {"labelFont": SANS, "titleFont": SANS, "labelFontSize": 11, "titleFontSize": 11,
                        "labelColor": INK_MUTED, "titleColor": INK_MUTED, "orient": "top", "symbolSize": 70},
            "range": {"category": SERIES},
            "header": {"labelFont": SANS, "titleFont": SANS, "labelFontSize": 11.5, "labelColor": INK},
        }
    }


def register_chart_theme() -> None:
    if hasattr(alt, "theme"):  # Altair >= 5.5
        alt.theme.register("stallm", enable=True)(_altair_theme)
    else:  # pragma: no cover - older Altair
        alt.themes.register("stallm", _altair_theme)
        alt.themes.enable("stallm")
