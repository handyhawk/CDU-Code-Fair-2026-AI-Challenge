"""
HumanFirst AI - frontend (Streamlit)

A calm, task-first staff portal for the triage backend, in the spirit of
public-service sites such as my.gov.au and GOV.UK: plain text, simple tables,
one main action per screen, and status that never relies on colour alone.

Backend contract used here (see backend/app.py):
    GET    /health                  -> status, local_model_trained
    POST   /triage                  -> analyse one message, returns the stored case
    (Batch upload reads the CSV here and calls /triage once per row, so it can
     show live progress.)
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
  --hf-line:#d5dce3; --hf-line-2:#e5eaef; --hf-bg:#ffffff; --hf-card:#ffffff; --hf-sunken:#f6f8fa;
  --hf-head:#eef2f6; --hf-hover:#f6f8fa; --hf-track:#e9edf1; --hf-heading:#12304f;
  --hf-banner:#12304f; --hf-input-line:#6b7a88; --hf-focus:#ffbf47;
  --hf-crit:#b3261e; --hf-crit-bg:#fbeceb; --hf-crit-ink:#8c1d18; --hf-crit-line:#e3a59e;
  --hf-high:#c77700; --hf-high-bg:#fdf1dc; --hf-high-ink:#7a4b00; --hf-high-line:#ecc88a;
  --hf-norm:#3f7d5a; --hf-norm-bg:#e7f1ea; --hf-norm-ink:#1f5a35; --hf-norm-line:#a9cdb6;
  --hf-info-bg:#eef4fa; --hf-info-line:#c3d4e5; --hf-info-ink:#26415d;
  --hf-chip-line:#cfd6dc; --hf-openai-ink:#12304f; --hf-disabled:#b8c3cf; --hf-sel:#eef4fa;
"""

DARK_VARS = """
  --hf-navy:#9cc3ea; --hf-navy-2:#5b8fc4; --hf-accent:#3d74ad; --hf-on-accent:#ffffff;
  --hf-ink:#e3e9ef; --hf-ink-2:#c9d3dc; --hf-muted:#93a3b3;
  --hf-line:#2d3b4a; --hf-line-2:#283544; --hf-bg:#0f1720; --hf-card:#16212d; --hf-sunken:#111b25;
  --hf-head:#1c2a38; --hf-hover:#1a2633; --hf-track:#243241; --hf-heading:#d6e4f2;
  --hf-banner:#0b2440; --hf-input-line:#5d6f82; --hf-focus:#ffbf47;
  --hf-crit:#ef6b62; --hf-crit-bg:#3a1a1a; --hf-crit-ink:#ffb4ac; --hf-crit-line:#7a2e29;
  --hf-high:#f0a43a; --hf-high-bg:#3a2a12; --hf-high-ink:#ffd08a; --hf-high-line:#7a5418;
  --hf-norm:#5fbf86; --hf-norm-bg:#15301f; --hf-norm-ink:#a6e3bd; --hf-norm-line:#2e6a45;
  --hf-info-bg:#15263a; --hf-info-line:#2c4a6b; --hf-info-ink:#b9d3ee;
  --hf-chip-line:#34465a; --hf-openai-ink:#b9d6f5; --hf-disabled:#3a4756; --hf-sel:#17283a;
"""

CSS = """
<style>
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]{display:none !important;}
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button, .stApp td, .stApp th,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp [data-testid="stMarkdownContainer"]{
  font-family:'Public Sans', 'Segoe UI', Arial, sans-serif;}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"]{background:var(--hf-bg) !important;}
[data-testid="stMain"]{container-type:inline-size;}
/* the <style> blocks render as empty elements; take them out of the layout */
.stElementContainer:has(style){display:none !important;}
.block-container{max-width:1100px; padding:0 2rem 0 !important;}
/* full-width bands that line up with the content column */
.hf-bleed{width:100cqw; margin-left:calc(50% - 50cqw);}
.hf-bleed > .in{max-width:calc(1100px - 4rem); margin:0 auto; padding:0 2rem; box-sizing:content-box;}
[data-testid="stHeaderActionElements"]{display:none !important;}

/* masthead */
.hf-mast{background:var(--hf-banner); color:#fff;}
.hf-mast .in{display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap;
  padding-top:16px; padding-bottom:16px;}
.hf-brand{display:flex; align-items:center; gap:12px;}
.hf-mark{width:36px; height:36px; border:2px solid #fff; display:flex; align-items:center; justify-content:center;
  font-weight:800; font-size:.85rem; letter-spacing:.02em;}
.hf-brand .t{font-size:1.2rem; font-weight:700; line-height:1.1;}
.hf-brand .s{font-size:.85rem; opacity:.85; margin-top:2px;}
.hf-mast .meta{font-size:.88rem;}
.hf-dot{display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:7px; background:#9aa7b4;}
.hf-dot.ok{background:#6fd09a;} .hf-dot.warn{background:#f0c05a;} .hf-dot.bad{background:#ff8a80;}

/* footer: one quiet line */
.hf-foot{border-top:1px solid var(--hf-line); margin-top:56px;}
.hf-foot .in{padding-top:16px; padding-bottom:28px; font-size:.84rem; color:var(--hf-muted);}

/* keyboard focus: high-visibility yellow, as on public-sector sites */
.stApp :focus-visible{outline:3px solid var(--hf-focus) !important; outline-offset:0 !important;}
/* form fields with a clear outline */
[data-testid="stTextAreaRootElement"], [data-testid="stTextInputRootElement"],
[data-testid="stSelectbox"] [role="group"], [data-testid="stMultiSelect"] [role="group"]{
  border:1.5px solid var(--hf-input-line) !important; border-radius:2px !important;}
.stApp button{border-radius:2px !important;}
button[kind="primary"]{font-weight:600 !important;}
h1,h2,h3{color:var(--hf-heading) !important; letter-spacing:-0.01em;}
h3{font-size:1.15rem !important; margin-top:.2rem;}

/* page heading */
.hf-intro{margin:30px 0 18px;}
.hf-intro h1{font-size:1.9rem !important; font-weight:700 !important; margin:0 !important; padding:0 !important;
  color:var(--hf-heading) !important; line-height:1.2 !important;}
.hf-intro p{margin:6px 0 0; color:var(--hf-muted) !important; font-size:1rem;}

/* urgency badge: the one coloured element; shape + word, never colour alone */
.hf-badge{display:inline-block; font-weight:700; font-size:.8rem; padding:3px 10px; border-radius:2px;
  border:1px solid; white-space:nowrap;}
.hf-badge.crit{background:var(--hf-crit-bg); color:var(--hf-crit-ink); border-color:var(--hf-crit-line);}
.hf-badge.high{background:var(--hf-high-bg); color:var(--hf-high-ink); border-color:var(--hf-high-line);}
.hf-badge.norm{background:var(--hf-norm-bg); color:var(--hf-norm-ink); border-color:var(--hf-norm-line);}
.hf-badge.lg{font-size:.95rem; padding:4px 13px;}
.hf-badge::before{content:""; display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:6px; vertical-align:1px;}
.hf-badge.crit::before{background:var(--hf-crit); border-radius:1px; transform:rotate(45deg);}
.hf-badge.high::before{background:var(--hf-high);}
.hf-badge.norm::before{background:transparent; border:2px solid var(--hf-norm); width:5px; height:5px;}

/* review state as plain text with a small marker */
.hf-st{display:inline-flex; align-items:center; white-space:nowrap; font-size:.9rem; font-weight:500; color:var(--hf-ink-2);}
.hf-st::before{content:""; display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:8px;}
.hf-st.need::before{background:var(--hf-high);}
.hf-st.done::before{background:var(--hf-norm);}
.hf-st.over::before{background:var(--hf-navy-2);}
.hf-st.none{color:var(--hf-muted); font-weight:400;} .hf-st.none::before{background:var(--hf-line);}
.hf-tag{margin-left:10px; font-size:.8rem; font-weight:600; color:var(--hf-high-ink);}
.hf-warn{color:var(--hf-high-ink); font-weight:600;}

/* case detail */
.hf-case{border:1px solid var(--hf-line); border-left:5px solid var(--hf-line); padding:18px 20px; margin-bottom:18px;
  background:var(--hf-card);}
.hf-case.crit{border-left-color:var(--hf-crit);} .hf-case.high{border-left-color:var(--hf-high);}
.hf-case.norm{border-left-color:var(--hf-norm);}
.hf-top{display:flex; gap:16px; align-items:center; flex-wrap:wrap;}
.hf-meta{font-size:.85rem; color:var(--hf-muted); margin:8px 0 0;}
.hf-quote{border-left:3px solid var(--hf-line); padding:2px 0 2px 14px; margin:16px 0 4px; font-size:1rem;
  color:var(--hf-ink); line-height:1.5;}
.hf-dl{display:grid; grid-template-columns:130px minmax(0,1fr); gap:10px 16px; margin:16px 0 0; font-size:.95rem;}
.hf-dl dt{color:var(--hf-muted); margin:0;} .hf-dl dd{color:var(--hf-ink); margin:0; line-height:1.45;}
@media (max-width:600px){.hf-dl{grid-template-columns:1fr; gap:2px 0;} .hf-dl dd{margin-bottom:10px;}}
.hf-note{border-left:3px solid; padding:6px 0 6px 12px; margin-top:16px; font-size:.9rem;}
.hf-note.warn{border-color:var(--hf-high); color:var(--hf-high-ink);}
.hf-note.info{border-color:var(--hf-navy-2); color:var(--hf-info-ink);}
.hf-record{border-left:3px solid var(--hf-norm); padding:4px 0 4px 14px; margin:2px 0 14px; font-size:.95rem; color:var(--hf-ink);}
.hf-record.over{border-color:var(--hf-navy-2);}
.hf-record .meta{font-size:.82rem; color:var(--hf-muted); margin-top:4px;}

/* tables */
.hf-table-wrap{border:1px solid var(--hf-line); background:var(--hf-card); max-height:340px; overflow:auto;}
table.hf-table{border-collapse:collapse; width:100%; font-size:.9rem; table-layout:fixed;}
table.hf-table th{position:sticky; top:0; background:var(--hf-head); color:var(--hf-ink-2); text-align:left; font-weight:700;
  padding:10px 12px; white-space:nowrap; border-bottom:1px solid var(--hf-line); font-size:.86rem;}
table.hf-table td{padding:10px 12px; border-bottom:1px solid var(--hf-line-2); vertical-align:middle; color:var(--hf-ink);
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
table.hf-table tr:last-child td{border-bottom:none;}
table.hf-table tr.sel td{background:var(--hf-sel);}
table.hf-table tr.sel td:first-child{box-shadow:inset 4px 0 0 var(--hf-navy-2);}
table.hf-table th:nth-child(1){width:58px;} table.hf-table th:nth-child(2){width:112px;}
table.hf-table th:nth-child(4){width:92px;} table.hf-table th:nth-child(5){width:112px;}
table.hf-table th:nth-child(6){width:250px;}
td.hf-id, td.hf-age, td.hf-eng{color:var(--hf-muted) !important;}
.hf-was{font-size:.72rem; color:var(--hf-muted); margin-top:2px;}

/* figures and bars */
.hf-stats{display:flex; flex-wrap:wrap; gap:8px 0; margin:4px 0 26px;}
.hf-stat{padding:0 36px 0 18px; border-left:3px solid var(--hf-line); min-width:120px;}
.hf-stat:first-child{padding-left:0; border-left:none;}
.hf-stat.crit{border-left-color:var(--hf-crit);} .hf-stat.high{border-left-color:var(--hf-high);}
.hf-stat .v{font-size:2rem; font-weight:700; color:var(--hf-heading); line-height:1.1;}
.hf-stat .k{font-size:.9rem; color:var(--hf-muted); margin-top:2px;}
.hf-panel{border-top:1px solid var(--hf-line); padding-top:12px; margin-bottom:26px;}
.hf-panel h4{margin:0 0 10px; font-size:1rem; color:var(--hf-heading);}
.hf-bar-row{display:grid; grid-template-columns:150px 1fr 40px; gap:10px; align-items:center; margin:8px 0; font-size:.92rem; color:var(--hf-ink);}
.hf-bar-track{background:var(--hf-track); height:10px;}
.hf-bar-fill{height:100%;}
.hf-bar-n{text-align:right; font-weight:600; color:var(--hf-ink);}
.hf-empty{border:1px dashed var(--hf-chip-line); padding:28px; text-align:center; color:var(--hf-muted);}
.hf-line{font-size:.95rem; color:var(--hf-ink-2); margin:6px 0 12px;}

/* native widgets */
button[kind="primary"]{background:var(--hf-accent) !important; border-color:var(--hf-accent) !important;}
button[kind="primary"], button[kind="primary"] *{color:var(--hf-on-accent) !important;}
button[kind="primary"]:hover{background:var(--hf-navy-2) !important;}
button[kind="primary"]:disabled{background:var(--hf-disabled) !important; border-color:var(--hf-disabled) !important; opacity:1;}
div[role="radiogroup"]{gap:8px;}
/* top navigation: plain tabs under the masthead, sticky while scrolling */
[data-testid="stHeader"]{background:transparent !important; pointer-events:none;}
[data-testid="stHeader"] *{pointer-events:auto;}
.st-key-page{width:100cqw !important; max-width:none !important; margin-left:calc(50% - 50cqw); background:var(--hf-card);
  margin-top:-1rem; border-bottom:1px solid var(--hf-line); padding:0; position:sticky; top:0; z-index:90;}
.st-key-page div[role="radiogroup"]{gap:0; flex-wrap:wrap; max-width:calc(1100px - 4rem); margin:0 auto; padding:0 2rem; box-sizing:content-box;}
.st-key-page [data-testid="stRadioOption"], .st-key-page label[data-baseweb="radio"]{background:transparent !important;
  border:none !important; border-radius:0; padding:14px 20px 11px 0; margin-right:12px; border-bottom:3px solid transparent !important;}
.st-key-page [data-testid="stRadioOption"] p, .st-key-page label[data-baseweb="radio"] p{color:var(--hf-ink-2) !important;
  font-weight:600; font-size:.98rem;}
.st-key-page [data-testid="stRadioOption"]:hover p{color:var(--hf-heading) !important;}
.st-key-page [data-testid="stRadioOption"][data-selected="true"], .st-key-page label[data-baseweb="radio"]:has(input:checked){
  border-bottom-color:var(--hf-navy-2) !important; background:transparent !important;}
.st-key-page [data-testid="stRadioOption"][data-selected="true"] p{color:var(--hf-heading) !important;}
/* other radios: quiet segmented choices */
[data-testid="stRadioOption"], label[data-baseweb="radio"]{background:var(--hf-card); border:1px solid var(--hf-line);
  border-radius:2px; padding:6px 16px; margin:0; cursor:pointer;}
[data-testid="stRadioOption"] > div > div:first-child, label[data-baseweb="radio"] > div:first-child{display:none;}
[data-testid="stRadioOption"] p, label[data-baseweb="radio"] p{font-weight:600; color:var(--hf-heading); margin:0;}
[data-testid="stRadioOption"][data-selected="true"], label[data-baseweb="radio"]:has(input:checked){
  background:var(--hf-accent); border-color:var(--hf-accent);}
[data-testid="stRadioOption"][data-selected="true"] p, label[data-baseweb="radio"]:has(input:checked) p{color:var(--hf-on-accent);}

[data-testid="stSidebar"], [data-testid="stExpandSidebarButton"]{display:none !important;}
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
[data-testid="stPopoverBody"]{background:var(--hf-card) !important; border:1px solid var(--hf-line) !important;}
[data-testid="stSelectboxVirtualDropdown"], [data-testid="stMultiSelectDropdown"]{background:var(--hf-card) !important;
  border:1px solid var(--hf-line) !important;}
[data-testid="stSelectboxVirtualDropdown"] *, [data-testid="stMultiSelectDropdown"] *{color:var(--hf-ink) !important;}
[data-testid="stMultiSelectDropdown"] [role="option"]:hover > *{background:var(--hf-head) !important;}
[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
[data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"],
[data-testid="stSelectboxVirtualDropdown"] [role="option"] > div{background-color:transparent;}
[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover > *,
[data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"] > *{background:var(--hf-head) !important;}

/* settings pop-up is rendered outside the app container, so theme its inputs directly */
[data-testid="stPopoverBody"] input{background:var(--hf-sunken) !important; color:var(--hf-ink) !important;
  -webkit-text-fill-color:var(--hf-ink) !important;}
[data-testid="stPopoverBody"] label, [data-testid="stPopoverBody"] p{color:var(--hf-ink) !important;}
[data-testid="stPopoverBody"] [data-testid="stTextInputRootElement"]{background:var(--hf-sunken) !important;}

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


def time_ago(iso_timestamp) -> str:
    try:
        created = datetime.fromisoformat(str(iso_timestamp).replace("Z", "+00:00"))
    except ValueError:
        return ""
    if created.tzinfo is None:
        created = created.astimezone()
    minutes = int((datetime.now().astimezone() - created).total_seconds() // 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes}m ago"
    if minutes < 60 * 24:
        return f"{minutes // 60}h ago"
    return f"{minutes // (60 * 24)}d ago"


def urgency_badge(urgency: str, large: bool = False) -> str:
    css = URGENCY_CLASS.get(urgency, "norm")
    return f'<span class="hf-badge {css}{" lg" if large else ""}">{esc(urgency)}</span>'


def engine_name(case: dict) -> str:
    return "OpenAI" if case.get("analysis_mode") == "OpenAI" else "Local ML"


def needs_review(case: dict) -> bool:
    return bool(case.get("escalate_to_human")) and not case.get("human_reviewed")


def current_urgency(case: dict) -> str:
    """Human decision where one exists, otherwise the AI's call."""
    if case.get("human_reviewed") and case.get("human_urgency"):
        return case["human_urgency"]
    return case.get("urgency", "Normal")


def was_overridden(case: dict) -> bool:
    return bool(case.get("human_reviewed")) and case.get("human_urgency") != case.get("urgency")


def review_state(case: dict) -> tuple:
    """(css class, plain-text label) for where the case is in the human review step."""
    if case.get("human_reviewed"):
        return ("over", "Overridden by reviewer") if was_overridden(case) else ("done", "Confirmed by reviewer")
    if case.get("escalate_to_human"):
        return "need", "Awaiting human review"
    return "none", "No review required"


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
    results, failed, skipped = [], [], 0
    started = datetime.now()

    for i, (row, text) in enumerate(messages, start=1):
        try:
            results.append(api("POST", "/triage", timeout=120, json={"message": text}))
        except requests.exceptions.ConnectionError as exc:
            # Backend is down: stop instead of failing every remaining row.
            failed.append((row, text, explain_error(exc)))
            skipped = len(messages) - i
            break
        except requests.exceptions.RequestException as exc:
            failed.append((row, text, explain_error(exc)))

        elapsed = (datetime.now() - started).total_seconds()
        remaining = elapsed / i * (total - i)
        eta = f" · about {remaining:.0f}s left" if i < total and i >= 2 else ""
        progress.progress(i / total, text=f"Analysed {i} of {total}{eta}")
        line = " · ".join(
            f"{sum(r.get('urgency') == level for r in results)} {level}" for level in URGENCY_ORDER
        )
        if failed:
            line += f" · {len(failed)} failed"
        counts_slot.markdown(f'<div class="hf-line">{esc(line)}</div>', unsafe_allow_html=True)

    progress.progress(1.0, text="Finished")
    return {"summary": {"total_messages": len(results)}, "results": results, "failed": failed,
            "skipped": skipped}


# ---------------------------------------------------------------------------
# HTML renderers
# ---------------------------------------------------------------------------

def render_header(health):
    """Masthead: service name on the left, service status on the right."""
    if health is None:
        status = '<span class="hf-dot bad"></span>Service offline'
    elif not health.get("local_model_trained"):
        status = '<span class="hf-dot warn"></span>Fallback model not ready'
    else:
        status = '<span class="hf-dot ok"></span>Service online'
    st.markdown(
        '<div class="hf-mast hf-bleed"><div class="in">'
        '<div class="hf-brand"><div class="hf-mark">HF</div><div><div class="t">HumanFirst AI</div>'
        '<div class="s">Inbox triage service</div></div></div>'
        f'<div class="meta">{status}</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )


def render_case(case: dict, show_message: bool = True):
    """One case, as plain facts: urgency, who analysed it, and where it is in human review."""
    urgency = case.get("urgency", "Normal")
    state_css, state_text = review_state(case)

    rows = [
        ("Category", esc(case.get("category") or "Unclassified")),
        ("Routed to", esc(case.get("route") or "Unclassified")),
    ]
    if case.get("analysis_mode") == "OpenAI":
        rows.append(("Analysed by", "OpenAI"))
    else:
        rows.append(("Analysed by", 'Local ML fallback <span class="hf-warn">&middot; Provisional</span>'))
        confidence = case.get("urgency_confidence")
        if confidence is not None:
            rows.append(("Confidence", f"{confidence:.0%}"))
    risks = case.get("matched_risk_keywords") or []
    if risks:
        rows.append(("Risk factors", esc(", ".join(map(str, risks)))))
    rows.append(("Explanation", esc(case.get("explanation") or "No explanation available.")))
    facts = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)

    notes = ""
    if case.get("provisional"):
        notes += (
            '<div class="hf-note warn"><b>Provisional.</b> OpenAI was unavailable, so the local model was used. '
            "A person must review this case before any action.</div>"
        )
    if case.get("attention_language_detected"):
        notes += (
            '<div class="hf-note info">Urgent-sounding wording was noted. It does not change the urgency.</div>'
        )
    safety = case.get("safety_notes")  # optional field, populated once the Safety Engine is added
    if safety:
        text = "; ".join(map(str, safety)) if isinstance(safety, list) else str(safety)
        notes += f'<div class="hf-note warn"><b>Safety check.</b> {esc(text)}</div>'

    quote = f'<div class="hf-quote">{multiline(case.get("message"))}</div>' if show_message else ""
    st.markdown(
        f'<div class="hf-case {URGENCY_CLASS.get(urgency, "norm")}">'
        f'<div class="hf-top">{urgency_badge(urgency, True)}<span class="hf-st {state_css}">{state_text}</span></div>'
        f'<div class="hf-meta">Case #{esc(case.get("id"))} &middot; Received {esc(fmt_time(case.get("created_at")))}</div>'
        f'{quote}<dl class="hf-dl">{facts}</dl>{notes}</div>',
        unsafe_allow_html=True,
    )


def status_cell(case: dict) -> str:
    state_css, state_text = review_state(case)
    cell = f'<span class="hf-st {state_css}">{state_text.replace(" by reviewer", "").replace("Awaiting human review", "Awaiting review")}</span>'
    if case.get("provisional") and not case.get("human_reviewed"):
        cell += '<span class="hf-tag" title="Local ML fallback result. A person must check it.">Provisional</span>'
    return cell


def render_table(cases: list, limit: int = 60, selected_id=None):
    """Compact, one line per case."""
    if not cases:
        return
    rows = ""
    for case in cases[:limit]:
        urgency = urgency_badge(current_urgency(case))
        if was_overridden(case):
            urgency += f'<div class="hf-was">AI: {esc(case.get("urgency"))}</div>'
        selected = ' class="sel"' if case.get("id") == selected_id else ""
        rows += (
            f'<tr{selected}><td class="hf-id">#{esc(case.get("id"))}</td><td>{urgency}</td>'
            f'<td title="{esc(case.get("message"))}">{esc(case.get("message"))}</td>'
            f'<td class="hf-age" title="{esc(fmt_time(case.get("created_at")))}">{esc(time_ago(case.get("created_at")))}</td>'
            f'<td class="hf-eng">{engine_name(case)}</td><td>{status_cell(case)}</td></tr>'
        )
    st.markdown(
        '<div class="hf-table-wrap"><table class="hf-table"><thead><tr><th>Case</th><th>Urgency</th>'
        "<th>Message</th><th>Received</th><th>Analysed by</th><th>Status</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>",
        unsafe_allow_html=True,
    )
    if len(cases) > limit:
        st.caption(f"Showing the first {limit} of {len(cases)} cases.")


def render_stats(items: list):
    """items: [(label, value, kind)] - plain figures, no boxes."""
    cells = "".join(
        f'<div class="hf-stat {kind}"><div class="v">{esc(value)}</div><div class="k">{esc(label)}</div></div>'
        for label, value, kind in items
    )
    st.markdown(f'<div class="hf-stats">{cells}</div>', unsafe_allow_html=True)


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


def render_intro(title: str, text: str = ""):
    """Page heading on the left, Settings on the right."""
    lede = f"<p>{esc(text)}</p>" if text else ""
    heading, settings = st.columns([5, 1], vertical_alignment="bottom")
    heading.markdown(f'<div class="hf-intro"><h1>{esc(title)}</h1>{lede}</div>', unsafe_allow_html=True)
    with settings:
        render_settings()


# ---------------------------------------------------------------------------
# Human review
# ---------------------------------------------------------------------------

def render_review(case: dict, prefix: str):
    """Human decision step. The AI recommends; the reviewer confirms or overrides (with a reason)."""
    case_id = case["id"]
    ai_urgency = case.get("urgency", "Normal")
    st.markdown("### Your decision")

    def review_form(default: str, default_notes: str, editing: bool):
        key = f"{prefix}_{case_id}"
        final = st.radio(
            "Final urgency", URGENCY_ORDER, horizontal=True, key=f"final_{key}",
            index=URGENCY_ORDER.index(default) if default in URGENCY_ORDER else 2,
            format_func=lambda u: f"{u}  (AI)" if u == ai_urgency else u,
        )
        overriding = final != ai_urgency
        notes = st.text_area(
            "Reason for override (required)" if overriding else "Notes (optional)",
            value=default_notes, key=f"notes_{key}", height=80,
            placeholder="Why does this case need a different urgency?" if overriding else "",
        )
        with st.expander("Draft reply (not sent automatically)"):
            st.text_area("Draft", value=ACK_TEMPLATES[final], height=110, key=f"ack_{key}_{final}",
                         label_visibility="collapsed")

        missing_reason = overriding and not notes.strip()
        if editing:
            label = "Update decision"
        else:
            label = f"Override to {final}" if overriding else f"Confirm {final}"
        if st.button(label, type="primary", key=f"save_{key}", disabled=missing_reason):
            try:
                updated = api(
                    "POST", f"/cases/{case_id}/review", timeout=15,
                    json={"human_urgency": final, "human_notes": notes.strip()},
                )
                verb = "overridden to" if final != ai_urgency else "confirmed as"
                st.session_state["flash"] = f"Case #{updated['id']} {verb} {final}."
                if st.session_state.get("last_result") and st.session_state["last_result"].get("id") == case_id:
                    st.session_state["last_result"] = updated
                st.rerun()
            except requests.exceptions.RequestException as exc:
                st.error(explain_error(exc))

    if case.get("human_reviewed"):
        overridden = was_overridden(case)
        note = f'<div>&ldquo;{esc(case["human_notes"])}&rdquo;</div>' if case.get("human_notes") else ""
        headline = (
            f"Overridden. The AI said {esc(ai_urgency)}; the reviewer set {esc(case.get('human_urgency'))}."
            if overridden else f"Confirmed as {esc(ai_urgency)}."
        )
        st.markdown(
            f'<div class="hf-record{" over" if overridden else ""}"><div><b>{headline}</b></div>{note}'
            f'<div class="meta">Recorded {esc(fmt_time(case.get("reviewed_at")))}</div></div>',
            unsafe_allow_html=True,
        )
        with st.expander("Change this decision"):
            review_form(case.get("human_urgency"), case.get("human_notes") or "", editing=True)
    else:
        if case.get("escalate_to_human"):
            st.caption("A person must confirm or override the AI before any action is taken.")
        review_form(ai_urgency, "", editing=False)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HumanFirst AI",
    page_icon="\U0001F4E8",
    layout="wide",
    initial_sidebar_state="collapsed",
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


def render_settings():
    """Small, out-of-the-way settings: theme, backend address, export and demo reset."""
    with st.popover("Settings", icon=":material/settings:", width="stretch"):
        st.toggle("Dark mode", key="dark_mode", on_change=_remember_theme)
        st.text_input("Backend address", key="backend_url")
        st.download_button(
            "Export all cases (CSV)", data=cases_to_csv(cases), disabled=not cases, width="stretch",
            file_name=f"humanfirst_cases_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv",
        )
        with st.expander("Reset demo data"):
            st.caption("Permanently deletes every stored case.")
            sure = st.checkbox("I understand this cannot be undone")
            if st.button("Delete all cases", disabled=not sure):
                try:
                    deleted = api("DELETE", "/cases", timeout=15)["deleted"]
                    st.session_state.update(last_result=None, last_batch=None, flash=f"Deleted {deleted} cases.")
                    st.rerun()
                except requests.exceptions.RequestException as exc:
                    st.error(explain_error(exc))


health = get_health()
cases, cases_error = get_cases()

PAGES = ["Triage a message", "Review queue", "Batch upload", "Dashboard"]
# Bumping this number gives the queue filters fresh keys, i.e. resets them to their defaults.
st.session_state.setdefault("q_filters_version", 0)

awaiting_all = sort_queue([c for c in cases if needs_review(c)])

render_header(health)

# Top navigation (a styled radio, so the chosen section survives the reruns after saving a decision).
page = st.radio(
    "Section", PAGES, horizontal=True, label_visibility="collapsed", key="page",
    format_func=lambda name: name + (f" ({len(awaiting_all)})" if name == PAGES[1] and awaiting_all else ""),
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
        "Start it with `python app.py` in the backend folder, or change the address in Settings."
    )


# ---- Triage ---------------------------------------------------------------
if page == PAGES[0]:
    render_intro("Triage a message")
    left, right = st.columns([2, 3], gap="large")

    def use_example():
        choice = st.session_state.get("example_choice")
        if choice in EXAMPLES:
            st.session_state["message_input"] = EXAMPLES[choice]

    def clear_message():
        st.session_state.update(message_input="", last_result=None, last_error=None,
                                example_choice="Choose an example...")

    with left:
        message = st.text_area(
            "Message", height=200, key="message_input",
            placeholder="Paste or type the message to analyse",
        )
        analyse, clear = st.columns([3, 2])
        if analyse.button("Analyse message", type="primary", disabled=not message.strip(), width="stretch"):
            with st.spinner("Analysing..."):
                try:
                    st.session_state["last_result"] = api("POST", "/triage", json={"message": message})
                    st.session_state["last_error"] = None
                except requests.exceptions.RequestException as exc:
                    st.session_state["last_error"] = explain_error(exc)
            st.rerun()
        clear.button("Clear", on_click=clear_message, width="stretch")
        st.selectbox(
            "Try an example",
            ["Choose an example..."] + list(EXAMPLES),
            key="example_choice",
            on_change=use_example,
        )

    with right:
        if st.session_state["last_error"]:
            st.error(st.session_state["last_error"])
        result = st.session_state["last_result"]
        if result:
            # show the freshest stored copy (picks up a saved human decision)
            result = next((c for c in cases if c.get("id") == result.get("id")), result)
            render_case(result, show_message=False)
            render_review(result, "triage")
        elif not st.session_state["last_error"]:
            render_empty("The assessment will appear here.")

# ---- Review queue -----------------------------------------------------------
if page == PAGES[1]:
    render_intro("Review queue")
    if cases_error:
        st.error(f"Could not load cases. {cases_error}")
    elif not cases:
        render_empty("No cases yet. Triage a message or upload a batch to start the queue.")
    else:
        v = st.session_state["q_filters_version"]
        counts = {
            "Awaiting review": len(awaiting_all),
            "Reviewed": sum(bool(c.get("human_reviewed")) for c in cases),
            "All cases": len(cases),
        }
        c_status, c_level = st.columns([3, 1], vertical_alignment="center")
        status = c_status.radio("Show", list(counts), horizontal=True, label_visibility="collapsed",
                                key=f"q_status_{v}", format_func=lambda s: f"{s} ({counts[s]})")
        level = c_level.selectbox("Urgency", ["All urgencies"] + URGENCY_ORDER, key=f"q_level_{v}",
                                  label_visibility="collapsed")

        shown = [
            c for c in cases
            if (level == "All urgencies" or current_urgency(c) == level)
            and (status == "All cases" or (status == "Reviewed") == bool(c.get("human_reviewed")))
            and (status != "Awaiting review" or needs_review(c))
        ]
        shown = sort_queue(shown)

        if not shown:
            render_empty("Nothing waiting for review." if status == "Awaiting review" else "No cases match.")
        else:
            by_id = {c["id"]: c for c in shown}
            # Open the most urgent case straight away; a selection that left the list falls back to the top.
            st.session_state.setdefault("q_case_version", 0)
            if st.session_state.get(f"queue_case_{st.session_state['q_case_version']}") not in by_id:
                st.session_state["q_case_version"] += 1  # fresh widget, so it starts on the top case
            case_key = f"queue_case_{st.session_state['q_case_version']}"
            chosen = st.session_state.get(case_key, shown[0]["id"])
            render_table(shown, selected_id=chosen)
            st.write("")
            chosen = st.selectbox(
                "Case to review", list(by_id), key=case_key,
                format_func=lambda i: f"#{i} · {current_urgency(by_id[i])} · {(by_id[i].get('message') or '')[:70]}",
            )
            render_case(by_id[chosen])
            render_review(by_id[chosen], "queue")

# ---- Batch -----------------------------------------------------------------
if page == PAGES[2]:
    render_intro("Batch upload", "Upload a CSV with a message column. Every row is analysed and saved.")
    uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    messages, read_error = read_batch_messages(uploaded) if uploaded else ([], None)
    if read_error:
        st.error(read_error)
    elif uploaded:
        st.caption(f"{len(messages)} message{'s' if len(messages) != 1 else ''} found in {uploaded.name}.")
    if messages and st.button("Analyse batch", type="primary"):
        st.session_state["last_batch"] = run_batch(messages)
        last = st.session_state["last_batch"]
        done, problems = len(last["results"]), len(last["failed"]) + last["skipped"]
        text = f"Analysed and saved {done} of {len(messages)} messages."
        if problems:
            st.session_state["flash_warning"] = text + " Some rows were not analysed; see the results below."
        else:
            st.session_state["flash"] = text
        st.rerun()

    batch = st.session_state["last_batch"]
    if batch:
        results = batch.get("results", [])
        st.markdown("### Results")
        if batch.get("skipped"):
            st.warning(
                f"The backend stopped responding, so the last {batch['skipped']} row(s) were not sent. "
                "Restart the backend and upload the file again; saved rows will be analysed a second time."
            )
        if batch.get("failed"):
            with st.expander(f"{len(batch['failed'])} message(s) could not be analysed"):
                for row, text, reason in batch["failed"]:
                    st.markdown(f"**Row {row}:** {esc(text[:120])}  \n<span style='color:var(--hf-crit)'>{esc(reason)}</span>",
                                unsafe_allow_html=True)
        stats = [
            ("Analysed", len(results), ""),
            ("Critical", sum(r.get("urgency") == "Critical" for r in results), "crit"),
            ("High", sum(r.get("urgency") == "High" for r in results), "high"),
            ("Normal", sum(r.get("urgency") == "Normal" for r in results), ""),
        ]
        provisional_n = sum(bool(r.get("provisional")) for r in results)
        if provisional_n:
            stats.append(("Provisional", provisional_n, ""))
        render_stats(stats)
        render_table(sort_queue(results))

# ---- Dashboard ---------------------------------------------------------------
if page == PAGES[3]:
    render_intro("Dashboard")
    if cases_error:
        st.error(f"Could not load cases. {cases_error}")
    elif not cases:
        render_empty("No data yet. Figures appear as messages are analysed.")
    else:
        now = [current_urgency(c) for c in cases]
        reviewed = [c for c in cases if c.get("human_reviewed")]
        overridden = [c for c in reviewed if was_overridden(c)]
        provisional = [c for c in cases if c.get("provisional")]
        render_stats([
            ("Total cases", len(cases), ""),
            ("Critical", now.count("Critical"), "crit"),
            ("High", now.count("High"), "high"),
            ("Awaiting review", len(awaiting_all), ""),
        ])

        c1, c2 = st.columns(2, gap="large")
        with c1:
            render_bars("Urgency", [(u, now.count(u), URGENCY_COLOR[u]) for u in URGENCY_ORDER])
            render_bars("Human review", [
                ("Awaiting review", len(awaiting_all), "var(--hf-high)"),
                ("Confirmed", len(reviewed) - len(overridden), "var(--hf-norm)"),
                ("Overridden", len(overridden), "var(--hf-navy-2)"),
                ("Not required", sum(not c.get("escalate_to_human") and not c.get("human_reviewed") for c in cases),
                 "var(--hf-muted)"),
            ])
        with c2:
            render_bars("Category", [(cat, sum(c.get("category") == cat for c in cases), "var(--hf-navy-2)")
                                     for cat in CATEGORIES])
            render_bars("Analysed by", [
                ("OpenAI", sum(c.get("analysis_mode") == "OpenAI" for c in cases), "var(--hf-navy-2)"),
                ("Local ML (provisional)", len(provisional), "var(--hf-muted)"),
            ])

st.markdown(
    '<div class="hf-foot hf-bleed"><div class="in">'
    "HumanFirst AI &middot; Prototype for the CDU IT Code Fair 2026 &middot; Synthetic messages only. "
    "The AI recommends; staff decide."
    "</div></div>",
    unsafe_allow_html=True,
)
