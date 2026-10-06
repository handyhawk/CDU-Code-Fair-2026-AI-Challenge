"""
HumanFirst AI - frontend (Streamlit)

Government / service-portal style interface for the triage backend.

Backend contract used here (see backend/app.py):
    GET    /health                  -> status, local_model_trained
    POST   /triage                  -> analyse one message, returns the stored case
    (Batch upload reads the CSV here and calls /triage once per row, so it can
     show live progress. /analyze still exists on the backend but is not used.)
    GET    /cases                   -> {"cases": [...]}  (newest first)
    POST   /cases/<id>/review       -> save the human decision
    DELETE /cases                   -> delete all stored cases

Every case carries: urgency, category, route, escalation, escalate_to_human,
analysis_mode ("OpenAI" / "Local ML"), provisional, explanation, human_* fields.
All text shown from the backend is HTML-escaped before it is rendered.
"""

import csv
import html
import io
from datetime import datetime

import requests
import streamlit as st

DEFAULT_BACKEND_URL = "http://localhost:5000"

URGENCY_ORDER = ["Critical", "High", "Normal"]
URGENCY_RANK = {name: i for i, name in enumerate(URGENCY_ORDER)}
URGENCY_CLASS = {"Critical": "crit", "High": "high", "Normal": "norm"}
URGENCY_COLOR = {"Critical": "var(--hf-crit)", "High": "var(--hf-high)", "Normal": "var(--hf-norm)"}

CATEGORIES = [
    "Health & Safety",
    "Housing & Utilities",
    "Financial Support",
    "Licensing & Services",
    "General Enquiries",
]

EXAMPLES = {
    "Power cut + insulin": (
        "My electricity was disconnected this morning and my insulin has to "
        "stay refrigerated. I don't know what to do."
    ),
    "Court deadline": (
        "I have a court hearing in three days and I'm still waiting for the "
        "documents I need from the department."
    ),
    "Routine enquiry": "Can someone confirm the opening hours for the service centre?",
    "Dramatic wording, routine": (
        "URGENT!!! I forgot my online portal password and I have zero memory "
        "of what I set it to \U0001F602"
    ),
}

# Drafts only - never sent automatically.
ACK_TEMPLATES = {
    "Critical": (
        "Thank you for contacting us. Your message has been identified as "
        "requiring urgent human review. A staff member will assess the "
        "information provided as soon as possible."
    ),
    "High": (
        "Thank you for your message. Your request has been marked for "
        "priority human review."
    ),
    "Normal": (
        "Thank you for contacting us. Your message has been received and "
        "will be reviewed through the standard process."
    ),
}

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

# Colour tokens. Every colour in CSS below comes from these, so switching the
# theme is just swapping one set of variables (plus a few native-widget fixes).
LIGHT_VARS = """
  --hf-navy:#12304f; --hf-navy-2:#1d466f; --hf-accent:#12304f; --hf-on-accent:#ffffff;
  --hf-ink:#1b2733; --hf-ink-2:#33414f; --hf-muted:#5b6b7b;
  --hf-line:#d5dce3; --hf-line-2:#e5eaef; --hf-bg:#f4f6f8; --hf-card:#ffffff; --hf-sunken:#f7f9fb;
  --hf-head:#eaeff4; --hf-hover:#f8fafc; --hf-track:#eceff2; --hf-heading:#12304f;
  --hf-banner:#12304f; --hf-banner-edge:#7fa3c7; --hf-nav:#1c4570;
  --hf-crit:#b3261e; --hf-crit-bg:#fbeceb; --hf-crit-ink:#8c1d18; --hf-crit-line:#e3a59e;
  --hf-high:#c77700; --hf-high-bg:#fdf1dc; --hf-high-ink:#7a4b00; --hf-high-line:#ecc88a;
  --hf-norm:#3f7d5a; --hf-norm-bg:#e7f1ea; --hf-norm-ink:#1f5a35; --hf-norm-line:#a9cdb6;
  --hf-info-bg:#eef4fa; --hf-info-line:#c3d4e5; --hf-info-ink:#26415d;
  --hf-chip-bg:#eef1f4; --hf-chip-ink:#3d4b59; --hf-chip-line:#cfd6dc;
  --hf-openai-bg:#e4edf6; --hf-openai-ink:#12304f; --hf-openai-line:#b7cbe0;
  --hf-disabled:#b8c3cf;
"""

DARK_VARS = """
  --hf-navy:#9cc3ea; --hf-navy-2:#5b8fc4; --hf-accent:#3d74ad; --hf-on-accent:#ffffff;
  --hf-ink:#e3e9ef; --hf-ink-2:#c9d3dc; --hf-muted:#93a3b3;
  --hf-line:#2d3b4a; --hf-line-2:#283544; --hf-bg:#0f1720; --hf-card:#16212d; --hf-sunken:#111b25;
  --hf-head:#1c2a38; --hf-hover:#1a2633; --hf-track:#243241; --hf-heading:#d6e4f2;
  --hf-banner:#0b2440; --hf-banner-edge:#3d74ad; --hf-nav:#123456;
  --hf-crit:#ef6b62; --hf-crit-bg:#3a1a1a; --hf-crit-ink:#ffb4ac; --hf-crit-line:#7a2e29;
  --hf-high:#f0a43a; --hf-high-bg:#3a2a12; --hf-high-ink:#ffd08a; --hf-high-line:#7a5418;
  --hf-norm:#5fbf86; --hf-norm-bg:#15301f; --hf-norm-ink:#a6e3bd; --hf-norm-line:#2e6a45;
  --hf-info-bg:#15263a; --hf-info-line:#2c4a6b; --hf-info-ink:#b9d3ee;
  --hf-chip-bg:#1e2b39; --hf-chip-ink:#c4cfda; --hf-chip-line:#34465a;
  --hf-openai-bg:#16304d; --hf-openai-ink:#b9d6f5; --hf-openai-line:#2f5680;
  --hf-disabled:#3a4756;
"""

CSS = """
<style>
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]{display:none !important;}
.block-container{max-width:1240px; padding-top:3rem; padding-bottom:3rem;}
h1,h2,h3{color:var(--hf-heading) !important; letter-spacing:-0.01em;}
h3{font-size:1.05rem !important; margin-top:.4rem;}

/* header */
.hf-header{background:var(--hf-banner); color:#fff; border-radius:6px 6px 0 0; padding:18px 24px;
  display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap;}
.hf-title{font-size:1.5rem; font-weight:700; line-height:1.2;}
.hf-sub{font-size:.9rem; opacity:.85; margin-top:2px;}
.hf-status{display:flex; gap:8px; flex-wrap:wrap;}
.hf-pill{font-size:.76rem; background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.28);
  border-radius:999px; padding:4px 11px; color:#fff; white-space:nowrap;}
.hf-dot{display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; background:#9aa7b4;}
.hf-dot.ok{background:#6fd09a;} .hf-dot.warn{background:#f0c05a;} .hf-dot.bad{background:#ff8a80;}

/* badges and chips */
.hf-badge{display:inline-block; font-weight:700; font-size:.78rem; padding:3px 10px; border-radius:4px;
  border:1px solid; letter-spacing:.02em; white-space:nowrap;}
.hf-badge.crit{background:var(--hf-crit-bg); color:var(--hf-crit-ink); border-color:var(--hf-crit-line);}
.hf-badge.high{background:var(--hf-high-bg); color:var(--hf-high-ink); border-color:var(--hf-high-line);}
.hf-badge.norm{background:var(--hf-norm-bg); color:var(--hf-norm-ink); border-color:var(--hf-norm-line);}
.hf-badge.lg{font-size:.95rem; padding:5px 14px;}
.hf-chip{display:inline-block; font-size:.74rem; padding:2px 9px; border-radius:999px; border:1px solid var(--hf-chip-line);
  color:var(--hf-chip-ink); background:var(--hf-chip-bg); white-space:nowrap; margin-right:4px;}
.hf-chip.openai{background:var(--hf-openai-bg); color:var(--hf-openai-ink); border-color:var(--hf-openai-line);}
.hf-chip.prov, .hf-chip.need{background:var(--hf-high-bg); color:var(--hf-high-ink); border-color:var(--hf-high-line); font-weight:600;}
.hf-chip.done{background:var(--hf-norm-bg); color:var(--hf-norm-ink); border-color:var(--hf-norm-line); font-weight:600;}
.hf-chip.none{background:var(--hf-sunken); color:var(--hf-muted);}
.hf-chip.risk{background:var(--hf-card); color:var(--hf-chip-ink);}
.hf-chip.over{background:var(--hf-info-bg); color:var(--hf-info-ink); border-color:var(--hf-info-line); font-weight:600;}
.hf-chip.local{background:var(--hf-chip-bg); color:var(--hf-chip-ink); border-color:var(--hf-chip-line); border-style:dashed;}
.hf-chip.openai{font-weight:600;}

/* status markers: shape + colour, so meaning never relies on colour alone */
.hf-badge::before, .hf-chip.openai::before, .hf-chip.local::before{content:""; display:inline-block; width:7px; height:7px;
  border-radius:50%; margin-right:6px; vertical-align:1px;}
.hf-badge.crit::before{background:var(--hf-crit); border-radius:1px; transform:rotate(45deg);}
.hf-badge.high::before{background:var(--hf-high);}
.hf-badge.norm::before{background:transparent; border:2px solid var(--hf-norm); width:5px; height:5px;}
.hf-chip.openai::before{background:var(--hf-openai-ink);}
.hf-chip.local::before{background:transparent; border:1.5px solid var(--hf-chip-ink); width:5px; height:5px;}
.hf-chip.prov::before{content:"!"; font-weight:800; margin-right:5px;}
.hf-chip.need::before{content:"\\25F7"; margin-right:5px;}
.hf-chip.done::before{content:"\\2713"; margin-right:5px; font-weight:800;}
.hf-chip.over::before{content:"\\21BA"; margin-right:5px; font-weight:800;}

/* page intro */
.hf-intro{margin:6px 0 14px;}
.hf-intro h2{font-size:1.3rem !important; margin:0 !important; padding:0 !important; color:var(--hf-heading) !important;}
.hf-intro p{margin:2px 0 0; color:var(--hf-muted) !important; font-size:.9rem;}

/* status key */
.hf-legend{display:flex; flex-wrap:wrap; gap:6px 18px; align-items:center; background:var(--hf-card);
  border:1px solid var(--hf-line); border-radius:6px; padding:10px 14px; margin:0 0 14px; font-size:.78rem; color:var(--hf-muted);}
.hf-legend .grp{display:flex; gap:6px; align-items:center; flex-wrap:wrap;}
.hf-legend .lbl{font-weight:700; text-transform:uppercase; letter-spacing:.05em; font-size:.66rem; margin-right:2px;}

/* human review decision panel */
.hf-decision{background:var(--hf-card); border:1px solid var(--hf-line); border-top:4px solid var(--hf-navy-2);
  border-radius:6px; padding:14px 16px; margin:4px 0 10px;}
.hf-decision .row{display:flex; gap:8px; align-items:center; flex-wrap:wrap; font-size:.88rem; color:var(--hf-ink);}
.hf-decision .muted{color:var(--hf-muted); font-size:.8rem; margin-top:6px;}
.hf-record{border-radius:6px; padding:12px 14px; border:1px solid; margin:4px 0 10px; font-size:.88rem;}
.hf-record.done{background:var(--hf-norm-bg); border-color:var(--hf-norm-line); color:var(--hf-norm-ink);}
.hf-record.over{background:var(--hf-info-bg); border-color:var(--hf-info-line); color:var(--hf-info-ink);}
.hf-record .t{font-weight:700; margin-bottom:6px;}
.hf-record .row{display:flex; gap:6px; align-items:center; flex-wrap:wrap; margin-bottom:4px;}
.hf-record .meta{font-size:.78rem; opacity:.85; margin-top:4px;}

/* case card */
.hf-case{background:var(--hf-card); border:1px solid var(--hf-line); border-left:6px solid var(--hf-line);
  border-radius:6px; padding:16px 18px; margin-bottom:12px;}
.hf-case.crit{border-left-color:var(--hf-crit);} .hf-case.high{border-left-color:var(--hf-high);}
.hf-case.norm{border-left-color:var(--hf-norm);}
.hf-eyebrow{font-size:.78rem; color:var(--hf-muted); margin-bottom:8px;}
.hf-badges{display:flex; gap:6px; flex-wrap:wrap; align-items:center; margin-bottom:12px;}
.hf-quote{background:var(--hf-sunken); border:1px solid var(--hf-line-2); border-radius:4px; padding:10px 12px;
  font-size:.92rem; color:var(--hf-ink); margin-bottom:12px; line-height:1.45;}
.hf-facts{display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin-bottom:12px;}
@media (max-width:900px){.hf-facts{grid-template-columns:repeat(2,minmax(0,1fr));}}
.hf-fact{border:1px solid var(--hf-line-2); border-radius:4px; padding:8px 10px; background:var(--hf-card);}
.hf-fact .k{font-size:.7rem; text-transform:uppercase; letter-spacing:.05em; color:var(--hf-muted);}
.hf-fact .v{font-size:.92rem; font-weight:600; color:var(--hf-ink); margin-top:2px;}
.hf-fact .sub{font-size:.7rem; color:var(--hf-muted); margin-top:2px; line-height:1.3;}
.hf-label{font-size:.7rem; text-transform:uppercase; letter-spacing:.05em; color:var(--hf-muted); margin:14px 0 4px;}
.hf-text{font-size:.92rem; color:var(--hf-ink); line-height:1.45;}
.hf-note{border-radius:4px; padding:9px 12px; font-size:.86rem; margin-top:10px; border:1px solid;}
.hf-note.warn{background:var(--hf-high-bg); border-color:var(--hf-high-line); color:var(--hf-high-ink);}
.hf-note.info{background:var(--hf-info-bg); border-color:var(--hf-info-line); color:var(--hf-info-ink);}
.hf-note.ok{background:var(--hf-norm-bg); border-color:var(--hf-norm-line); color:var(--hf-norm-ink);}

/* tables */
.hf-table-wrap{border:1px solid var(--hf-line); border-radius:6px; background:var(--hf-card); max-height:470px; overflow:auto;}
table.hf-table{border-collapse:collapse; width:100%; font-size:.84rem;}
table.hf-table th{position:sticky; top:0; background:var(--hf-head); color:var(--hf-ink-2); text-align:left; font-weight:700;
  padding:9px 10px; border-bottom:1px solid var(--hf-line); font-size:.74rem; text-transform:uppercase; letter-spacing:.04em;}
table.hf-table td{padding:9px 10px; border-bottom:1px solid var(--hf-line-2); vertical-align:middle; color:var(--hf-ink);}
table.hf-table tr:hover td{background:var(--hf-hover);}
td.hf-msg{min-width:260px; color:var(--hf-ink-2) !important;}
table.hf-table td:nth-child(1), table.hf-table td:nth-child(2){white-space:nowrap;}

/* KPI + bars */
[data-testid="stHeaderActionElements"]{display:none !important;}
.hf-kpis{display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:12px; margin:6px 0 16px;}
.hf-kpi{background:var(--hf-card); border:1px solid var(--hf-line); border-top:4px solid var(--hf-navy-2); border-radius:6px; padding:12px 14px;}
.hf-kpi.crit{border-top-color:var(--hf-crit);} .hf-kpi.high{border-top-color:var(--hf-high);}
.hf-kpi.norm{border-top-color:var(--hf-norm);}
.hf-kpi .k{font-size:.72rem; text-transform:uppercase; letter-spacing:.05em; color:var(--hf-muted);}
.hf-kpi .v{font-size:1.9rem; font-weight:700; color:var(--hf-heading); line-height:1.15;}
.hf-kpi .s{font-size:.78rem; color:var(--hf-muted);}
.hf-panel{background:var(--hf-card); border:1px solid var(--hf-line); border-radius:6px; padding:14px 16px; margin-bottom:12px;}
.hf-panel h4{margin:0 0 10px; font-size:.95rem; color:var(--hf-heading);}
.hf-bar-row{display:grid; grid-template-columns:150px 1fr 40px; gap:10px; align-items:center; margin:7px 0; font-size:.85rem; color:var(--hf-ink);}
.hf-bar-track{background:var(--hf-track); border-radius:3px; height:12px; overflow:hidden;}
.hf-bar-fill{height:100%; border-radius:3px;}
.hf-bar-n{text-align:right; font-weight:600; color:var(--hf-ink);}
.hf-empty{border:1px dashed var(--hf-chip-line); border-radius:6px; padding:28px; text-align:center; color:var(--hf-muted); background:var(--hf-card);}
.hf-footer{margin-top:28px; padding-top:12px; border-top:1px solid var(--hf-line); font-size:.78rem; color:var(--hf-muted);}

/* native widgets */
button[kind="primary"]{background:var(--hf-accent) !important; border-color:var(--hf-accent) !important;}
button[kind="primary"], button[kind="primary"] *{color:var(--hf-on-accent) !important;}
button[kind="primary"]:hover{background:var(--hf-navy-2) !important;}
button[kind="primary"]:disabled{background:var(--hf-disabled) !important; border-color:var(--hf-disabled) !important; opacity:1;}
div[role="radiogroup"]{gap:8px;}
/* top navigation bar, attached under the banner and sticky while scrolling */
[data-testid="stHeader"]{background:transparent !important; pointer-events:none;}
[data-testid="stHeader"] *{pointer-events:auto;}
.st-key-page{width:100% !important; background:var(--hf-nav); margin-top:-1rem; border-radius:0 0 6px 6px; padding:0 12px;
  border-bottom:4px solid var(--hf-banner-edge); position:sticky; top:0; z-index:90;
  box-shadow:0 2px 6px rgba(0,0,0,.08);}
.st-key-page div[role="radiogroup"]{gap:2px; flex-wrap:wrap;}
.st-key-page [data-testid="stRadioOption"], .st-key-page label[data-baseweb="radio"]{background:transparent !important;
  border:none !important; border-radius:0; padding:13px 18px 11px; border-bottom:3px solid transparent !important;}
.st-key-page [data-testid="stRadioOption"] p, .st-key-page label[data-baseweb="radio"] p{color:rgba(255,255,255,.78) !important;
  font-weight:600; font-size:.92rem;}
.st-key-page [data-testid="stRadioOption"] svg, .st-key-page [data-testid="stRadioOption"] [data-testid="stIconMaterial"]{
  color:rgba(255,255,255,.78) !important;}
.st-key-page [data-testid="stRadioOption"]:hover{background:rgba(255,255,255,.08) !important;}
.st-key-page [data-testid="stRadioOption"]:hover p{color:#fff !important;}
.st-key-page [data-testid="stRadioOption"][data-selected="true"], .st-key-page label[data-baseweb="radio"]:has(input:checked){
  background:rgba(255,255,255,.12) !important; border-bottom-color:#ffffff !important;}
.st-key-page [data-testid="stRadioOption"][data-selected="true"] p,
.st-key-page [data-testid="stRadioOption"][data-selected="true"] [data-testid="stIconMaterial"]{color:#fff !important;}
[data-testid="stRadioOption"], label[data-baseweb="radio"]{background:var(--hf-card); border:1px solid var(--hf-line);
  border-radius:4px; padding:6px 16px; margin:0; cursor:pointer;}
[data-testid="stRadioOption"] > div > div:first-child, label[data-baseweb="radio"] > div:first-child{display:none;}
[data-testid="stRadioOption"] p, label[data-baseweb="radio"] p{font-weight:600; color:var(--hf-heading); margin:0;}
[data-testid="stRadioOption"][data-selected="true"], label[data-baseweb="radio"]:has(input:checked){
  background:var(--hf-accent); border-color:var(--hf-accent);}
[data-testid="stRadioOption"][data-selected="true"] p, label[data-baseweb="radio"]:has(input:checked) p{color:var(--hf-on-accent);}
.hf-toggle-row{display:flex; justify-content:flex-end;}

/* sidebar */
[data-testid="stSidebar"]{border-right:1px solid var(--hf-line);}
[data-testid="stSidebarHeader"]{height:2.6rem; padding-top:.6rem; padding-bottom:0; margin-bottom:0;}
[data-testid="stSidebarUserContent"]{padding-top:0 !important;}
.hf-side-brand{display:flex; align-items:center; gap:10px; padding-bottom:12px; margin-bottom:6px; border-bottom:1px solid var(--hf-line);}
.hf-side-logo{width:34px; height:34px; border-radius:8px; background:var(--hf-banner); color:#fff; font-weight:800;
  display:flex; align-items:center; justify-content:center; font-size:.9rem; letter-spacing:.02em;
  border-bottom:3px solid var(--hf-banner-edge);}
.hf-side-name{font-weight:700; color:var(--hf-heading); line-height:1.1;}
.hf-side-tag{font-size:.72rem; color:var(--hf-muted);}
.hf-side-h{font-size:.68rem; text-transform:uppercase; letter-spacing:.08em; color:var(--hf-muted); font-weight:700; margin:16px 0 6px;}
.hf-side-stat{display:flex; justify-content:space-between; align-items:center; padding:7px 10px; margin-bottom:5px;
  background:var(--hf-sunken); border:1px solid var(--hf-line-2); border-left:4px solid var(--hf-line); border-radius:5px;
  font-size:.83rem; color:var(--hf-ink);}
.hf-side-stat b{font-size:1rem; color:var(--hf-heading);}
.hf-side-stat.crit{border-left-color:var(--hf-crit);} .hf-side-stat.high{border-left-color:var(--hf-high);}
.hf-side-stat.norm{border-left-color:var(--hf-norm);}
.hf-side-sys{font-size:.8rem; color:var(--hf-ink); line-height:1.9;}
.hf-side-sys .hf-dot{vertical-align:middle;}
.hf-side-sys .muted{color:var(--hf-muted); font-size:.72rem;}
.hf-side-foot{font-size:.72rem; color:var(--hf-muted); margin-top:18px; padding-top:10px; border-top:1px solid var(--hf-line); line-height:1.5;}
</style>
"""

# Extra rules for dark mode only: Streamlit's own widgets are themed by
# .streamlit/config.toml (light), so they need to be recoloured here.
DARK_WIDGET_CSS = """
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"]{background:var(--hf-bg) !important;}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div{background:var(--hf-card) !important;}
.stApp, .stApp p, .stApp li, .stApp label, .stApp span, [data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"] *{color:var(--hf-ink);}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *{color:var(--hf-muted) !important;}
.stApp code{background:var(--hf-head) !important; color:var(--hf-openai-ink) !important;}
.stApp a{color:var(--hf-navy) !important;}

/* text inputs, text areas, select boxes */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"], [data-baseweb="select"] > div{
  background:var(--hf-sunken) !important; border-color:var(--hf-line) !important;}
.stApp input, .stApp textarea{background:var(--hf-sunken) !important; color:var(--hf-ink) !important;
  -webkit-text-fill-color:var(--hf-ink) !important; caret-color:var(--hf-ink);}
.stApp input::placeholder, .stApp textarea::placeholder{color:var(--hf-muted) !important; -webkit-text-fill-color:var(--hf-muted) !important;}
[data-baseweb="select"] *{color:var(--hf-ink) !important;}
[data-baseweb="select"] svg{fill:var(--hf-muted) !important;}
[data-baseweb="popover"] ul, [data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"]{background:var(--hf-card) !important;}
[data-baseweb="popover"] li{background:var(--hf-card) !important; color:var(--hf-ink) !important;}
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] li[aria-selected="true"]{background:var(--hf-head) !important;}
/* Streamlit 1.64 widget markup */
body{background:var(--hf-bg) !important; color:var(--hf-ink);}
[data-testid="stTextAreaRootElement"], [data-testid="stTextInputRootElement"], [data-testid="stNumberInputContainer"],
[data-testid="stSelectbox"] [role="group"], [data-testid="stMultiSelect"] [role="group"]{
  background:var(--hf-sunken) !important; border-color:var(--hf-line) !important;}
[data-testid="stTextAreaRootElement"]:focus-within, [data-testid="stTextInputRootElement"]:focus-within,
[data-testid="stSelectbox"] [role="group"]:focus-within, [data-testid="stMultiSelect"] [role="group"]:focus-within{
  border-color:var(--hf-navy-2) !important;}
[data-testid="stSelectbox"] [role="group"] *{color:var(--hf-ink) !important;}
[data-testid="stMultiSelect"] [role="group"] input{background:transparent !important;}
[data-testid="stSelectbox"] [role="group"] svg, [data-testid="stMultiSelect"] [role="group"] > div:last-child svg{
  color:var(--hf-muted) !important;}
[data-testid="stSelectboxVirtualDropdown"], [data-testid="stMultiSelectDropdown"]{background:var(--hf-card) !important;
  border:1px solid var(--hf-line) !important;}
[data-testid="stSelectboxVirtualDropdown"] *, [data-testid="stMultiSelectDropdown"] *{color:var(--hf-ink) !important;}
[data-testid="stMultiSelectDropdown"] [role="option"]:hover > *{background:var(--hf-head) !important;}
[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
[data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"],
[data-testid="stSelectboxVirtualDropdown"] [role="option"] > div{background-color:transparent;}
[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover > *,
[data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"] > *{background:var(--hf-head) !important;}

/* secondary buttons */
button[kind="secondary"]{background:var(--hf-card) !important; border-color:var(--hf-line) !important;}
button[kind="secondary"], button[kind="secondary"] *{color:var(--hf-ink) !important;}
button[kind="secondary"]:hover{border-color:var(--hf-navy-2) !important;}

/* file uploader */
[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploader"] section{background:var(--hf-card) !important;
  border-color:var(--hf-line) !important;}
[data-testid="stFileUploaderFile"], [data-testid="stFileChip"]{background:var(--hf-head) !important;}
[data-testid="stFileUploader"] *{color:var(--hf-ink);}
[data-testid="stFileUploader"] small{color:var(--hf-muted) !important;}

/* expanders */
[data-testid="stExpander"] details{background:var(--hf-card) !important; border-color:var(--hf-line) !important;}
[data-testid="stExpander"] summary{background:var(--hf-head) !important;}
[data-testid="stExpander"] summary *{color:var(--hf-ink) !important;}
[data-testid="stExpander"] svg{fill:var(--hf-ink) !important;}

/* alerts (success / warning / error / info) */
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]){background:var(--hf-norm-bg) !important; color:var(--hf-norm-ink) !important;}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]){background:var(--hf-high-bg) !important; color:var(--hf-high-ink) !important;}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]){background:var(--hf-crit-bg) !important; color:var(--hf-crit-ink) !important;}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]){background:var(--hf-info-bg) !important; color:var(--hf-info-ink) !important;}
[data-testid="stAlertContainer"] *{color:inherit !important;}
[data-testid="stAlertContainer"] a{text-decoration:underline;}
[data-testid="stAlertContainer"] code{background:rgba(255,255,255,.08) !important;}

/* progress bar, toggle, tooltips */
[data-testid="stProgress"] > div > div > div{background:var(--hf-track) !important;}
[data-testid="stProgress"] > div > div > div > div{background:var(--hf-navy-2) !important;}
[data-testid="stTooltipIcon"] svg{stroke:var(--hf-muted) !important;}
</style>
"""


def theme_css(dark: bool) -> str:
    variables = DARK_VARS if dark else LIGHT_VARS
    scheme = "dark" if dark else "light"
    css = f"<style>:root{{{variables} color-scheme:{scheme};}}</style>" + CSS
    return css + DARK_WIDGET_CSS if dark else css


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def multiline(value) -> str:
    return esc(value).replace("\n", "<br>")


def fmt_time(iso_timestamp) -> str:
    try:
        return datetime.fromisoformat(iso_timestamp).astimezone().strftime("%d %b, %H:%M")
    except (TypeError, ValueError):
        return ""


def urgency_badge(urgency: str, large: bool = False) -> str:
    css = URGENCY_CLASS.get(urgency, "norm")
    return f'<span class="hf-badge {css}{" lg" if large else ""}">{esc(urgency)}</span>'


def mode_chip(case: dict) -> str:
    mode = case.get("analysis_mode") or "Unknown"
    if mode == "OpenAI":
        return '<span class="hf-chip openai" title="Classified by the primary OpenAI model">OpenAI</span>'
    return (
        '<span class="hf-chip local" title="OpenAI was unavailable, so the local fallback model was used">'
        f"{esc(mode)} fallback</span>"
    )


def provisional_chip(case: dict) -> str:
    if not case.get("provisional"):
        return ""
    return (
        '<span class="hf-chip prov" title="Fallback result. A person must check it before any action.">'
        "Provisional</span>"
    )


def needs_review(case: dict) -> bool:
    return bool(case.get("escalate_to_human")) and not case.get("human_reviewed")


def review_chip(case: dict) -> str:
    if case.get("human_reviewed"):
        if was_overridden(case):
            return '<span class="hf-chip over">Human override</span>'
        return '<span class="hf-chip done">Human confirmed</span>'
    if case.get("escalate_to_human"):
        return '<span class="hf-chip need">Awaiting human review</span>'
    return '<span class="hf-chip none">No review required</span>'


def current_urgency(case: dict) -> str:
    """Human decision where one exists, otherwise the AI's call."""
    if case.get("human_reviewed") and case.get("human_urgency"):
        return case["human_urgency"]
    return case.get("urgency", "Normal")


def was_overridden(case: dict) -> bool:
    return bool(case.get("human_reviewed")) and case.get("human_urgency") != case.get("urgency")


def confidence_text(case: dict) -> str:
    """OpenAI gives no calibrated probability, so show N/A rather than a fake number."""
    if case.get("analysis_mode") == "OpenAI":
        return "N/A"
    value = case.get("urgency_confidence")
    return "N/A" if value is None else f"{value:.0%}"


def sort_queue(cases: list) -> list:
    """Most urgent first; within a level, the case waiting longest comes first."""
    return sorted(
        cases,
        key=lambda c: (URGENCY_RANK.get(current_urgency(c), 9), c.get("created_at") or ""),
    )


def cases_to_csv(cases: list) -> str:
    fields = [
        "id", "created_at", "message", "urgency", "human_urgency", "urgency_confidence",
        "category", "route", "escalation", "escalate_to_human", "analysis_mode", "provisional",
        "matched_risk_keywords", "source", "human_reviewed", "human_notes", "reviewed_at",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for case in cases:
        row = dict(case)
        if isinstance(row.get("matched_risk_keywords"), list):
            row["matched_risk_keywords"] = ", ".join(row["matched_risk_keywords"])
        writer.writerow(row)
    return buffer.getvalue()


def waiting_since(cases: list) -> str:
    """How long the oldest case awaiting review has been waiting, e.g. '2h 15m'."""
    times = []
    for case in cases:
        try:
            created = datetime.fromisoformat(str(case.get("created_at")).replace("Z", "+00:00"))
        except ValueError:
            continue
        if created.tzinfo is None:
            created = created.astimezone()
        times.append(created)
    if not times:
        return "-"
    minutes = int((datetime.now().astimezone() - min(times)).total_seconds() // 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes}m"
    if minutes < 60 * 24:
        return f"{minutes // 60}h {minutes % 60}m"
    return f"{minutes // (60 * 24)}d {minutes // 60 % 24}h"


# ---------------------------------------------------------------------------
# Backend calls
# ---------------------------------------------------------------------------

def api(method: str, path: str, timeout: int = 30, **kwargs):
    response = requests.request(
        method, f"{st.session_state['backend_url'].rstrip('/')}{path}", timeout=timeout, **kwargs
    )
    response.raise_for_status()
    return response.json()


def get_health():
    try:
        return api("GET", "/health", timeout=5)
    except requests.exceptions.RequestException:
        return None


def get_cases():
    try:
        return api("GET", "/cases", timeout=15)["cases"], None
    except requests.exceptions.RequestException as exc:
        return [], str(exc)


def explain_error(exc: Exception) -> str:
    if isinstance(exc, requests.exceptions.ConnectionError):
        return f"Could not reach the backend at {st.session_state['backend_url']}. Is app.py running?"
    if isinstance(exc, requests.exceptions.Timeout):
        return "The backend took too long to respond. Try again."
    return f"The backend returned an error: {exc}"


# Same column names the backend accepts in data_loader.COLUMN_ALIASES.
MESSAGE_COLUMN_ALIASES = {"message", "text", "message_text", "content"}


def read_batch_messages(uploaded):
    """Read the uploaded CSV in the browser session. Returns (messages, error)."""
    import pandas as pd  # installed with Streamlit

    raw = uploaded.getvalue()
    df = None
    for encoding in ("utf-8-sig", "cp1252"):  # cp1252 covers CSVs saved from Excel on Windows
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except Exception as exc:  # empty file, malformed rows, ...
            return [], f"Could not read this CSV: {exc}"
    if df is None:
        return [], "Could not read this CSV. Save it as UTF-8 and try again."

    column = next(
        (c for c in df.columns if str(c).strip().lower().replace(" ", "_") in MESSAGE_COLUMN_ALIASES), None
    )
    if column is None:
        return [], f"The CSV needs a `message` (or `text`) column. Found: {', '.join(map(str, df.columns))}"

    messages = []
    for index, value in df[column].items():
        if pd.isna(value) or not str(value).strip():
            continue
        messages.append((index + 2, str(value).strip()))  # +2 = spreadsheet row (header is row 1)
    if not messages:
        return [], "The message column is empty."
    return messages, None


def run_batch(messages):
    """Send each message to /triage one at a time, showing live progress."""
    total = len(messages)
    progress = st.progress(0.0, text=f"Starting... 0 of {total}")
    counts_slot = st.empty()
    latest_slot = st.empty()
    results, failed, skipped = [], [], 0
    started = datetime.now()

    for i, (row, text) in enumerate(messages, start=1):
        try:
            case = api("POST", "/triage", timeout=120, json={"message": text})
            results.append(case)
            latest = (f"Row {row} → <b>{esc(case.get('urgency', '?'))}</b> "
                      f"({esc(case.get('analysis_mode', ''))}): {esc(text[:90])}")
        except requests.exceptions.ConnectionError as exc:
            # Backend is down: stop instead of failing every remaining row.
            failed.append((row, text, explain_error(exc)))
            skipped = len(messages) - i
            break
        except requests.exceptions.RequestException as exc:
            failed.append((row, text, explain_error(exc)))
            latest = f"Row {row} failed: {esc(text[:90])}"

        elapsed = (datetime.now() - started).total_seconds()
        remaining = elapsed / i * (total - i)
        eta = f" · about {remaining:.0f}s left" if i < total and i >= 2 else ""
        progress.progress(i / total, text=f"Analysed {i} of {total}{eta}")
        with counts_slot.container():
            render_kpis([
                ("Done", f"{i}/{total}", "", ""),
                ("Critical", sum(r.get("urgency") == "Critical" for r in results), "", "crit"),
                ("High", sum(r.get("urgency") == "High" for r in results), "", "high"),
                ("Normal", sum(r.get("urgency") == "Normal" for r in results), "", "norm"),
                ("Failed", len(failed), "", ""),
            ])
        latest_slot.markdown(f"<div style='color:var(--hf-muted);font-size:.9rem'>{latest}</div>",
                             unsafe_allow_html=True)

    progress.progress(1.0, text="Finished")
    return {"summary": {"total_messages": len(results)}, "results": results, "failed": failed,
            "skipped": skipped}


# ---------------------------------------------------------------------------
# HTML renderers
# ---------------------------------------------------------------------------

def render_header(health):
    if health is None:
        backend = '<span class="hf-pill"><span class="hf-dot bad"></span>Backend unreachable</span>'
        rest = ""
    else:
        backend = '<span class="hf-pill"><span class="hf-dot ok"></span>Backend connected</span>'
        trained = health.get("local_model_trained")
        rest = (
            '<span class="hf-pill"><span class="hf-dot ok"></span>Primary: OpenAI</span>'
            + (
                '<span class="hf-pill"><span class="hf-dot ok"></span>Fallback: Local ML ready</span>'
                if trained
                else '<span class="hf-pill"><span class="hf-dot warn"></span>Fallback: Local ML not trained</span>'
            )
        )
    st.markdown(
        '<div class="hf-header"><div><div class="hf-title">HumanFirst AI</div>'
        '<div class="hf-sub">Inbox triage decision support &middot; the AI assists, staff decide</div></div>'
        f'<div class="hf-status">{backend}{rest}</div></div>',
        unsafe_allow_html=True,
    )


def render_case(case: dict):
    urgency = case.get("urgency", "Normal")
    facts = [
        ("Category", case.get("category") or "Unclassified"),
        ("Routed to", case.get("route") or "Unclassified"),
        ("Escalation tier", case.get("escalation") or "No"),
        ("Model confidence", confidence_text(case)),
    ]
    facts_html = "".join(
        f'<div class="hf-fact"><div class="k">{k}</div><div class="v">{esc(v)}</div>'
        + ('<div class="sub">OpenAI gives no calibrated probability</div>'
           if k == "Model confidence" and case.get("analysis_mode") == "OpenAI" else "")
        + "</div>"
        for k, v in facts
    )

    risks = case.get("matched_risk_keywords") or []
    risks_html = ""
    if risks:
        chips = "".join(f'<span class="hf-chip risk">{esc(r)}</span>' for r in risks)
        risks_html = f'<div class="hf-label">Risk factors identified</div><div>{chips}</div>'

    notes = ""
    if case.get("provisional"):
        notes += (
            '<div class="hf-note warn"><b>Provisional result.</b> The OpenAI classifier was unavailable, '
            "so the local fallback model was used. A staff member must review this case before any action.</div>"
        )
    if case.get("attention_language_detected"):
        notes += (
            '<div class="hf-note info">Urgent-sounding language was detected. On its own this does '
            "not change the urgency level.</div>"
        )
    safety = case.get("safety_notes")  # optional field, populated once the Safety Engine is added
    if safety:
        text = "; ".join(map(str, safety)) if isinstance(safety, list) else str(safety)
        notes += f'<div class="hf-note warn"><b>Safety Engine.</b> {esc(text)}</div>'

    st.markdown(
        f'<div class="hf-case {URGENCY_CLASS.get(urgency, "norm")}">'
        f'<div class="hf-eyebrow">Case #{esc(case.get("id"))} &middot; Received {esc(fmt_time(case.get("created_at")))}</div>'
        f'<div class="hf-badges">{urgency_badge(urgency, True)}{mode_chip(case)}{provisional_chip(case)}{review_chip(case)}</div>'
        f'<div class="hf-quote">{multiline(case.get("message"))}</div>'
        f'<div class="hf-facts">{facts_html}</div>'
        f'<div class="hf-label">Explanation</div><div class="hf-text">{esc(case.get("explanation") or "No explanation available.")}</div>'
        f"{risks_html}{notes}</div>",
        unsafe_allow_html=True,
    )


def render_table(cases: list, limit: int = 60):
    if not cases:
        return
    rows = ""
    for case in cases[:limit]:
        shown = urgency_badge(case.get("urgency", "Normal"))
        if was_overridden(case):
            shown += f' &rarr; {urgency_badge(case["human_urgency"])}'
        message = case.get("message") or ""
        short = message if len(message) <= 110 else message[:107] + "..."
        rows += (
            f'<tr><td>#{esc(case.get("id"))}</td><td>{esc(fmt_time(case.get("created_at")))}</td>'
            f"<td>{shown}</td><td>{esc(case.get('category'))}</td><td>{esc(case.get('route'))}</td>"
            f"<td>{mode_chip(case)}{provisional_chip(case)}</td><td>{review_chip(case)}</td>"
            f'<td class="hf-msg">{esc(short)}</td></tr>'
        )
    st.markdown(
        '<div class="hf-table-wrap"><table class="hf-table"><thead><tr><th>ID</th><th>Received</th>'
        "<th>Urgency</th><th>Category</th><th>Routed to</th><th>Analysed by</th><th>Review</th><th>Message</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>",
        unsafe_allow_html=True,
    )
    if len(cases) > limit:
        st.caption(f"Showing the first {limit} of {len(cases)} matching cases.")


def render_kpis(items: list):
    cards = "".join(
        f'<div class="hf-kpi {kind}"><div class="k">{esc(label)}</div><div class="v">{esc(value)}</div>'
        f'<div class="s">{esc(sub)}</div></div>'
        for label, value, sub, kind in items
    )
    st.markdown(f'<div class="hf-kpis">{cards}</div>', unsafe_allow_html=True)


def render_bars(title: str, items: list):
    """items: [(label, count, colour)]"""
    top = max([n for _, n, _ in items] + [1])
    rows = "".join(
        f'<div class="hf-bar-row"><div>{esc(label)}</div><div class="hf-bar-track">'
        f'<div class="hf-bar-fill" style="width:{n / top * 100:.0f}%;background:{color}"></div></div>'
        f'<div class="hf-bar-n">{n}</div></div>'
        for label, n, color in items
    )
    st.markdown(f'<div class="hf-panel"><h4>{esc(title)}</h4>{rows}</div>', unsafe_allow_html=True)


def render_empty(text: str):
    st.markdown(f'<div class="hf-empty">{esc(text)}</div>', unsafe_allow_html=True)


def render_intro(title: str, text: str):
    st.markdown(f'<div class="hf-intro"><h2>{esc(title)}</h2><p>{esc(text)}</p></div>', unsafe_allow_html=True)


def render_legend():
    """A one-line key so anyone watching the demo can read the labels."""
    st.markdown(
        '<div class="hf-legend">'
        '<div class="grp"><span class="lbl">Urgency</span>'
        f'{urgency_badge("Critical")}{urgency_badge("High")}{urgency_badge("Normal")}</div>'
        '<div class="grp"><span class="lbl">Analysed by</span>'
        '<span class="hf-chip openai">OpenAI</span><span class="hf-chip local">Local ML fallback</span>'
        '<span class="hf-chip prov">Provisional</span></div>'
        '<div class="grp"><span class="lbl">Human review</span>'
        '<span class="hf-chip need">Awaiting human review</span><span class="hf-chip done">Human confirmed</span>'
        '<span class="hf-chip over">Human override</span></div>'
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Human review
# ---------------------------------------------------------------------------

def render_review(case: dict, prefix: str, next_case_id=None):
    """Human decision step. The AI recommends; the reviewer confirms or overrides (with a reason)."""
    case_id = case["id"]
    ai_urgency = case.get("urgency", "Normal")
    st.markdown("### Human review")

    def review_form(default: str, default_notes: str, editing: bool):
        key = f"{prefix}_{case_id}"
        final = st.radio(
            "Final urgency", URGENCY_ORDER, horizontal=True, key=f"final_{key}",
            index=URGENCY_ORDER.index(default) if default in URGENCY_ORDER else 2,
            format_func=lambda u: f"{u}  (AI)" if u == ai_urgency else u,
            help="Starts on the AI recommendation, marked (AI). Pick another level to override it.",
        )
        overriding = final != ai_urgency
        notes = st.text_area(
            "Reason for override (required)" if overriding else "Reviewer notes (optional)",
            value=default_notes, key=f"notes_{key}", height=80,
            placeholder="Why does this case need a different urgency?" if overriding
            else "Anything the next person should know",
        )
        with st.expander("Draft acknowledgement (not sent automatically)"):
            st.text_area("Draft", value=ACK_TEMPLATES[final], height=110, key=f"ack_{key}_{final}")
            st.caption("A draft only. Staff decide whether and how to use it.")

        missing_reason = overriding and not notes.strip()
        if editing:
            label = "Update decision"
        else:
            label = f"Override to {final}" if overriding else f"Confirm {final}"
        if missing_reason:
            st.caption("Add a reason to override the AI recommendation. It is kept with the case for accountability.")
        if st.button(label, type="primary", key=f"save_{key}", disabled=missing_reason):
            try:
                updated = api(
                    "POST", f"/cases/{case_id}/review", timeout=15,
                    json={"human_urgency": final, "human_notes": notes.strip()},
                )
                verb = "overridden to" if final != ai_urgency else "confirmed as"
                flash = f"Case #{updated['id']} {verb} {final}."
                if prefix == "queue" and next_case_id and not editing:
                    st.session_state["pending_case"] = next_case_id
                    flash += f" Opened the next waiting case, #{next_case_id}."
                st.session_state["flash"] = flash
                if st.session_state.get("last_result") and st.session_state["last_result"].get("id") == case_id:
                    st.session_state["last_result"] = updated
                st.rerun()
            except requests.exceptions.RequestException as exc:
                st.error(explain_error(exc))

    if case.get("human_reviewed"):
        overridden = was_overridden(case)
        note = f'<div>&ldquo;{esc(case["human_notes"])}&rdquo;</div>' if case.get("human_notes") else ""
        change = (
            f"AI recommended {urgency_badge(ai_urgency)} &rarr; reviewer set {urgency_badge(case.get('human_urgency'))}"
            if overridden else f"Reviewer confirmed the AI recommendation {urgency_badge(ai_urgency)}"
        )
        st.markdown(
            f'<div class="hf-record {"over" if overridden else "done"}">'
            f'<div class="t">{"Decision recorded: human override" if overridden else "Decision recorded: confirmed"}</div>'
            f'<div class="row">{change}</div>{note}'
            f'<div class="meta">Recorded {esc(fmt_time(case.get("reviewed_at")))}</div></div>',
            unsafe_allow_html=True,
        )
        with st.expander("Change this decision"):
            review_form(case.get("human_urgency"), case.get("human_notes") or "", editing=True)
    else:
        required = case.get("escalate_to_human")
        st.markdown(
            '<div class="hf-decision"><div class="row">AI recommendation: '
            f"{urgency_badge(ai_urgency)}{mode_chip(case)}{provisional_chip(case)}</div>"
            '<div class="muted">'
            + ("Review required before any action. Confirm the recommendation or override it."
               if required else "Review is optional for this case. You can still confirm or override it.")
            + "</div></div>",
            unsafe_allow_html=True,
        )
        review_form(ai_urgency, "", editing=False)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HumanFirst AI",
    page_icon="\U0001F4E8",
    layout="wide",
    initial_sidebar_state="expanded",
)
def _theme_from_url() -> bool:
    # ?theme=light / ?theme=dark in the address bar keeps the choice across page refreshes.
    return st.query_params.get("theme", "dark") != "light"


def _remember_theme():
    st.query_params["theme"] = "dark" if st.session_state["dark_mode"] else "light"


st.session_state.setdefault("dark_mode", _theme_from_url())
st.markdown(theme_css(st.session_state["dark_mode"]), unsafe_allow_html=True)

for key, default in {
    "backend_url": DEFAULT_BACKEND_URL,
    "last_result": None,
    "last_error": None,
    "last_batch": None,
    "message_input": "",
    "flash": None,
}.items():
    st.session_state.setdefault(key, default)

with st.sidebar:
    # Settings first: the URL must be known before the backend is contacted below.
    side_top = st.container()
    side_bottom = st.container()
    with side_bottom:
        with st.expander("Connection settings"):
            st.session_state["backend_url"] = st.text_input("Backend URL", value=st.session_state["backend_url"])
            st.caption("Cases are stored by the backend and persist across restarts.")

health = get_health()

cases, cases_error = get_cases()

PAGES = ["Triage a message", "Review queue", "Batch upload", "Dashboard"]
# Bumping this number gives the queue filters fresh keys, i.e. resets them to their defaults.
st.session_state.setdefault("q_filters_version", 0)


def go_to(page_name: str):
    st.session_state["page"] = page_name


def new_message():
    st.session_state.update(page=PAGES[0], message_input="", last_result=None, last_error=None,
                            example_choice="Choose an example...")


def review_next(case_id: int):
    # Reset the queue filters so the chosen case is guaranteed to be in the list.
    st.session_state["q_filters_version"] += 1
    st.session_state.update(page=PAGES[1], pending_case=case_id)


awaiting_all = sort_queue([c for c in cases if needs_review(c)])
with side_top:
    st.markdown(
        '<div class="hf-side-brand"><div class="hf-side-logo">HF</div><div>'
        '<div class="hf-side-name">HumanFirst AI</div><div class="hf-side-tag">Inbox triage console</div></div></div>',
        unsafe_allow_html=True,
    )
    st.toggle("Dark mode", key="dark_mode", on_change=_remember_theme,
              help="Switch between dark and light colours. Your choice is kept if you refresh the page.")

    st.markdown('<div class="hf-side-h">Inbox at a glance</div>', unsafe_allow_html=True)
    critical_waiting = sum(current_urgency(c) == "Critical" for c in awaiting_all)
    st.markdown(
        f'<div class="hf-side-stat crit">Critical waiting <b>{critical_waiting}</b></div>'
        f'<div class="hf-side-stat high">Awaiting review <b>{len(awaiting_all)}</b></div>'
        f'<div class="hf-side-stat">Oldest waiting <b>{esc(waiting_since(awaiting_all))}</b></div>'
        f'<div class="hf-side-stat norm">Total cases <b>{len(cases)}</b></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="hf-side-h">Quick actions</div>', unsafe_allow_html=True)
    if awaiting_all:
        top = awaiting_all[0]
        st.button(f"Review next case (#{top['id']}, {current_urgency(top)})", width="stretch", type="primary",
                  on_click=review_next, args=(top["id"],), icon=":material/assignment_ind:")
    else:
        st.button("No cases waiting", width="stretch", disabled=True, icon=":material/task_alt:")
    st.button("New message", width="stretch", on_click=new_message, icon=":material/edit_note:")
    st.button("Upload a batch", width="stretch", on_click=go_to, args=(PAGES[2],), icon=":material/upload_file:")
    st.button("Refresh data", width="stretch", icon=":material/refresh:")  # any click reruns the app
    st.download_button(
        "Export all cases (CSV)", data=cases_to_csv(cases), width="stretch", disabled=not cases,
        file_name=f"humanfirst_cases_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv",
        icon=":material/download:",
    )

    st.markdown('<div class="hf-side-h">System status</div>', unsafe_allow_html=True)
    if health is None:
        sys_rows = '<span class="hf-dot bad"></span>Backend unreachable'
    else:
        trained = health.get("local_model_trained")
        sys_rows = (
            '<span class="hf-dot ok"></span>Backend online<br>'
            f'<span class="hf-dot ok"></span>Primary: {esc(health.get("primary_model", "OpenAI"))}<br>'
            f'<span class="hf-dot {"ok" if trained else "warn"}"></span>Fallback: '
            f'{esc(health.get("fallback_model", "Local ML"))} {"ready" if trained else "not trained"}'
        )
    st.markdown(
        f'<div class="hf-side-sys">{sys_rows}<br><span class="muted">Last checked {datetime.now():%H:%M:%S}</span></div>',
        unsafe_allow_html=True,
    )

with side_bottom:
    st.markdown(
        '<div class="hf-side-foot"><b>The AI assists, staff decide.</b> Every Critical or High case '
        "goes to a person before any action.<br>CDU IT Code Fair 2026 &middot; Team HumanFirst</div>",
        unsafe_allow_html=True,
    )

render_header(health)

# Top navigation bar (a styled radio, so the chosen section survives the reruns after saving a decision).
NAV_ICONS = {
    PAGES[0]: ":material/edit_note:",
    PAGES[1]: ":material/fact_check:",
    PAGES[2]: ":material/upload_file:",
    PAGES[3]: ":material/monitoring:",
}
page = st.radio(
    "Section", PAGES, horizontal=True, label_visibility="collapsed", key="page",
    format_func=lambda name: f"{NAV_ICONS[name]} {name}"
    + (f" ({len(awaiting_all)})" if name == PAGES[1] and awaiting_all else ""),
)

if st.session_state["flash"]:
    st.success(st.session_state["flash"])
    st.session_state["flash"] = None
if st.session_state.get("flash_warning"):
    st.warning(st.session_state["flash_warning"])
    st.session_state["flash_warning"] = None

if health is None:
    st.error(
        f"The backend at {st.session_state['backend_url']} is not reachable. "
        "Start it with `python app.py` in the backend folder, or change it under Connection settings in the sidebar."
    )


# ---- Triage ---------------------------------------------------------------
if page == PAGES[0]:
    render_intro("Triage a message",
                 "Paste an incoming message. The AI recommends an urgency, category and team; a person makes the decision.")
    left, right = st.columns([2, 3], gap="large")

    def use_example():
        choice = st.session_state.get("example_choice")
        if choice in EXAMPLES:
            st.session_state["message_input"] = EXAMPLES[choice]

    with left:
        st.markdown("### Incoming message")
        st.selectbox(
            "Load an example (for demos)",
            ["Choose an example..."] + list(EXAMPLES),
            key="example_choice",
            on_change=use_example,
        )
        message = st.text_area(
            "Message text", height=190, key="message_input",
            placeholder="Paste or type the message to analyse...",
        )
        if st.button("Analyse message", type="primary", disabled=not message.strip()):
            with st.spinner("Analysing..."):
                try:
                    st.session_state["last_result"] = api("POST", "/triage", json={"message": message})
                    st.session_state["last_error"] = None
                except requests.exceptions.RequestException as exc:
                    st.session_state["last_error"] = explain_error(exc)
            st.rerun()
        st.caption(
            "Messages are classified by OpenAI. If it is unavailable, the local model is used and the "
            "result is marked provisional."
        )

    with right:
        st.markdown("### Assessment")
        if st.session_state["last_error"]:
            st.error(st.session_state["last_error"])
        result = st.session_state["last_result"]
        if result:
            # show the freshest stored copy (picks up a saved human decision)
            result = next((c for c in cases if c.get("id") == result.get("id")), result)
            render_case(result)
            render_review(result, "triage")
        elif not st.session_state["last_error"]:
            render_empty("Analyse a message to see its urgency, category, routing and explanation here.")

# ---- Review queue -----------------------------------------------------------
if page == PAGES[1]:
    render_intro("Review queue",
                 "Cases waiting for a person, most urgent first. Open a case to confirm or override the AI.")
    if cases_error:
        st.error(f"Could not load cases. {cases_error}")
    elif not cases:
        render_empty("No cases yet. Analyse a message or upload a CSV to start the queue.")
    else:
        awaiting = [c for c in cases if needs_review(c)]
        render_kpis([
            ("Awaiting human review", len(awaiting), "Critical and High cases not yet decided", "high"),
            ("Critical waiting", sum(current_urgency(c) == "Critical" for c in awaiting), "Highest priority", "crit"),
            ("Provisional (Local ML)", sum(bool(c.get("provisional")) and not c.get("human_reviewed") for c in cases),
             "Fallback results to verify", ""),
            ("Reviewed", sum(bool(c.get("human_reviewed")) for c in cases), f"of {len(cases)} cases", "norm"),
        ])

        render_legend()
        v = st.session_state["q_filters_version"]
        f1, f2, f3 = st.columns(3)
        status = f1.selectbox("Status", ["Awaiting review", "Reviewed", "All cases"], key=f"q_status_{v}")
        levels = f2.multiselect("Urgency", URGENCY_ORDER, default=URGENCY_ORDER, key=f"q_levels_{v}")
        modes = f3.multiselect("Analysed by", ["OpenAI", "Local ML"], default=["OpenAI", "Local ML"],
                               key=f"q_modes_{v}")

        shown = [
            c for c in cases
            if current_urgency(c) in levels
            and (c.get("analysis_mode") or "Local ML") in modes
            and (status == "All cases" or (status == "Reviewed") == bool(c.get("human_reviewed")))
            and (status != "Awaiting review" or needs_review(c))
        ]
        shown = sort_queue(shown)
        st.caption("Most urgent first. Within the same level, the case that has waited longest is at the top.")

        if not shown:
            render_empty("No cases match these filters.")
        else:
            render_table(shown)
            st.markdown("### Open a case")
            by_id = {c["id"]: c for c in shown}
            options = [0] + list(by_id)
            pending = st.session_state.pop("pending_case", None)
            if pending in by_id:
                st.session_state["queue_case"] = pending
            chosen = st.selectbox(
                "Case", options, key="queue_case",
                format_func=lambda i: "Select a case..." if i == 0
                else f"#{i} · {current_urgency(by_id[i])} · {(by_id[i].get('message') or '')[:70]}",
            )
            if chosen in by_id:
                waiting_ids = [c["id"] for c in sort_queue([c for c in cases if needs_review(c)]) if c["id"] != chosen]
                render_case(by_id[chosen])
                render_review(by_id[chosen], "queue", next_case_id=waiting_ids[0] if waiting_ids else None)

# ---- Batch -----------------------------------------------------------------
if page == PAGES[2]:
    render_intro("Batch upload", "Triage a whole inbox export at once. Every row is saved to the case store.")
    st.write(
        "Upload a CSV with a `message` (or `text`) column. Each row goes through the same pipeline as a "
        "single message."
    )
    uploaded = st.file_uploader("CSV file", type=["csv"])
    messages, read_error = read_batch_messages(uploaded) if uploaded else ([], None)
    if read_error:
        st.error(read_error)
    elif uploaded:
        st.caption(
            f"Found **{len(messages)}** message{'s' if len(messages) != 1 else ''} in {esc(uploaded.name)}. "
            "Each one is analysed separately, so you'll see progress row by row."
        )
    if messages and st.button("Analyse batch", type="primary"):
        st.session_state["last_batch"] = run_batch(messages)
        last = st.session_state["last_batch"]
        done, problems = len(last["results"]), len(last["failed"]) + last["skipped"]
        text = f"Analysed and saved {done} of {len(messages)} messages."
        if problems:
            st.session_state["flash_warning"] = text + " Some rows were not analysed; see Last batch below."
        else:
            st.session_state["flash"] = text
        st.rerun()

    batch = st.session_state["last_batch"]
    if batch:
        results = batch.get("results", [])
        summary = batch.get("summary", {})
        st.markdown("### Last batch")
        if batch.get("skipped"):
            st.warning(
                f"The backend stopped responding, so the last {batch['skipped']} row(s) were not sent. "
                "Restart the backend and upload the file again; already-saved rows will be analysed a second time, "
                "so remove them from the CSV first if you don't want duplicates."
            )
        if batch.get("failed"):
            with st.expander(f"{len(batch['failed'])} message(s) could not be analysed"):
                for row, text, reason in batch["failed"]:
                    st.markdown(f"**Row {row}:** {esc(text[:120])}  \n<span style='color:var(--hf-crit)'>{esc(reason)}</span>",
                                unsafe_allow_html=True)
        render_kpis([
            ("Messages", summary.get("total_messages", len(results)), "analysed and saved", ""),
            ("Critical", sum(r.get("urgency") == "Critical" for r in results), "", "crit"),
            ("High", sum(r.get("urgency") == "High" for r in results), "", "high"),
            ("Normal", sum(r.get("urgency") == "Normal" for r in results), "", "norm"),
            ("Provisional", sum(bool(r.get("provisional")) for r in results), "Local ML fallback used", ""),
        ])
        render_table(sort_queue(results))

# ---- Dashboard ---------------------------------------------------------------
if page == PAGES[3]:
    render_intro("Dashboard", "How the inbox is being handled: urgency mix, review progress and how often staff override the AI.")
    if cases_error:
        st.error(f"Could not load cases. {cases_error}")
    elif not cases:
        render_empty("No data yet. The dashboard fills as messages are analysed.")
    else:
        total = len(cases)
        now = [current_urgency(c) for c in cases]
        reviewed = [c for c in cases if c.get("human_reviewed")]
        overridden = [c for c in reviewed if was_overridden(c)]
        provisional = [c for c in cases if c.get("provisional")]
        render_kpis([
            ("Total cases", total, "all time", ""),
            ("Critical", now.count("Critical"), "current urgency", "crit"),
            ("High", now.count("High"), "current urgency", "high"),
            ("Awaiting review", sum(needs_review(c) for c in cases), "need a human decision", "high"),
            ("Human overrides", len(overridden),
             f"{len(overridden) / len(reviewed):.0%} of reviewed" if reviewed else "no reviews yet", ""),
            ("Fallback used", len(provisional), f"{len(provisional) / total:.0%} of cases", ""),
        ])

        c1, c2 = st.columns(2, gap="large")
        with c1:
            render_bars("Urgency (human decision where reviewed)",
                        [(u, now.count(u), URGENCY_COLOR[u]) for u in URGENCY_ORDER])
            render_bars("Analysis source", [
                ("OpenAI", sum(c.get("analysis_mode") == "OpenAI" for c in cases), "var(--hf-navy-2)"),
                ("Local ML fallback", len(provisional), "var(--hf-muted)"),
            ])
        with c2:
            render_bars("Category", [(cat, sum(c.get("category") == cat for c in cases), "var(--hf-navy-2)") for cat in CATEGORIES])
            render_bars("Review status", [
                ("Awaiting review", sum(needs_review(c) for c in cases), "var(--hf-high)"),
                ("Reviewed", len(reviewed), "var(--hf-norm)"),
                ("No review required", sum(not c.get("escalate_to_human") and not c.get("human_reviewed") for c in cases), "var(--hf-muted)"),
            ])

        st.markdown("### Recent activity")
        render_table(cases[:10])

        st.markdown("### Export and administration")
        e1, e2 = st.columns([1, 2])
        e1.download_button(
            "Export cases as CSV", data=cases_to_csv(cases),
            file_name=f"humanfirst_cases_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv",
        )
        with e2.expander("Delete all stored cases"):
            st.warning("This permanently deletes every stored case. Use it only to reset demo data.")
            sure = st.checkbox("I understand this cannot be undone")
            if st.button("Delete all cases", disabled=not sure):
                try:
                    deleted = api("DELETE", "/cases", timeout=15)["deleted"]
                    st.session_state.update(last_result=None, last_batch=None, flash=f"Deleted {deleted} cases.")
                    st.rerun()
                except requests.exceptions.RequestException as exc:
                    st.error(explain_error(exc))

st.markdown(
    '<div class="hf-footer">Decision-support prototype for the CDU IT Code Fair 2026 AI Challenge. '
    "Uses synthetic data. The AI recommends; authorised staff make the final decision.</div>",
    unsafe_allow_html=True,
)
