"""
HumanFirst AI - frontend (Streamlit)

Government / service-portal style interface for the triage backend.

Backend contract used here (see backend/app.py):
    GET    /health                  -> status, local_model_trained
    POST   /triage                  -> analyse one message, returns the stored case
    POST   /analyze                 -> analyse a CSV, returns {"summary", "results"}
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
URGENCY_COLOR = {"Critical": "#b3261e", "High": "#c77700", "Normal": "#3f7d5a"}

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

CSS = """
<style>
:root{
  --hf-navy:#12304f; --hf-navy-2:#1d466f; --hf-ink:#1b2733; --hf-muted:#5b6b7b;
  --hf-line:#d5dce3; --hf-bg:#f4f6f8; --hf-card:#ffffff;
  --hf-crit:#b3261e; --hf-crit-bg:#fbeceb; --hf-crit-ink:#8c1d18;
  --hf-high:#c77700; --hf-high-bg:#fdf1dc; --hf-high-ink:#7a4b00;
  --hf-norm:#3f7d5a; --hf-norm-bg:#e7f1ea; --hf-norm-ink:#1f5a35;
}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]{display:none !important;}
.block-container{max-width:1240px; padding-top:3rem; padding-bottom:3rem;}
h1,h2,h3{color:var(--hf-navy); letter-spacing:-0.01em;}
h3{font-size:1.05rem !important; margin-top:.4rem;}

/* header */
.hf-header{background:var(--hf-navy); color:#fff; border-radius:6px; padding:18px 24px;
  display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap;
  border-bottom:4px solid #7fa3c7; margin-bottom:14px;}
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
.hf-badge.crit{background:var(--hf-crit-bg); color:var(--hf-crit-ink); border-color:#e3a59e;}
.hf-badge.high{background:var(--hf-high-bg); color:var(--hf-high-ink); border-color:#ecc88a;}
.hf-badge.norm{background:var(--hf-norm-bg); color:var(--hf-norm-ink); border-color:#a9cdb6;}
.hf-badge.lg{font-size:.95rem; padding:5px 14px;}
.hf-chip{display:inline-block; font-size:.74rem; padding:2px 9px; border-radius:999px; border:1px solid var(--hf-line);
  color:#3d4b59; background:#eef1f4; white-space:nowrap; margin-right:4px;}
.hf-chip.openai{background:#e4edf6; color:var(--hf-navy); border-color:#b7cbe0;}
.hf-chip.local{background:#eceff1; color:#44525f; border-color:#cfd6dc;}
.hf-chip.prov{background:#fff7e6; color:#7a4b00; border-color:#e8c37a; font-weight:600;}
.hf-chip.need{background:#fff7e6; color:#7a4b00; border-color:#e8c37a; font-weight:600;}
.hf-chip.done{background:var(--hf-norm-bg); color:var(--hf-norm-ink); border-color:#a9cdb6; font-weight:600;}
.hf-chip.none{background:#f4f6f8; color:#66747f;}
.hf-chip.risk{background:#fff; color:#3d4b59; border-color:#b9c3cc;}

/* case card */
.hf-case{background:var(--hf-card); border:1px solid var(--hf-line); border-left:6px solid var(--hf-line);
  border-radius:6px; padding:16px 18px; margin-bottom:12px;}
.hf-case.crit{border-left-color:var(--hf-crit);} .hf-case.high{border-left-color:var(--hf-high);}
.hf-case.norm{border-left-color:var(--hf-norm);}
.hf-eyebrow{font-size:.78rem; color:var(--hf-muted); margin-bottom:8px;}
.hf-badges{display:flex; gap:6px; flex-wrap:wrap; align-items:center; margin-bottom:12px;}
.hf-quote{background:#f7f9fb; border:1px solid #e5eaef; border-radius:4px; padding:10px 12px;
  font-size:.92rem; color:var(--hf-ink); margin-bottom:12px; line-height:1.45;}
.hf-facts{display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin-bottom:12px;}
@media (max-width:900px){.hf-facts{grid-template-columns:repeat(2,minmax(0,1fr));}}
.hf-fact{border:1px solid #e5eaef; border-radius:4px; padding:8px 10px; background:#fff;}
.hf-fact .k{font-size:.7rem; text-transform:uppercase; letter-spacing:.05em; color:var(--hf-muted);}
.hf-fact .v{font-size:.92rem; font-weight:600; color:var(--hf-ink); margin-top:2px;}
.hf-label{font-size:.7rem; text-transform:uppercase; letter-spacing:.05em; color:var(--hf-muted); margin:14px 0 4px;}
.hf-text{font-size:.92rem; color:var(--hf-ink); line-height:1.45;}
.hf-note{border-radius:4px; padding:9px 12px; font-size:.86rem; margin-top:10px; border:1px solid;}
.hf-note.warn{background:#fff7e6; border-color:#e8c37a; color:#6b4300;}
.hf-note.info{background:#eef4fa; border-color:#c3d4e5; color:#26415d;}
.hf-note.ok{background:var(--hf-norm-bg); border-color:#a9cdb6; color:var(--hf-norm-ink);}

/* tables */
.hf-table-wrap{border:1px solid var(--hf-line); border-radius:6px; background:#fff; max-height:470px; overflow:auto;}
table.hf-table{border-collapse:collapse; width:100%; font-size:.84rem;}
table.hf-table th{position:sticky; top:0; background:#eaeff4; color:#2d3c4b; text-align:left; font-weight:700;
  padding:9px 10px; border-bottom:1px solid var(--hf-line); font-size:.74rem; text-transform:uppercase; letter-spacing:.04em;}
table.hf-table td{padding:9px 10px; border-bottom:1px solid #edf0f3; vertical-align:middle; color:var(--hf-ink);}
table.hf-table tr:hover td{background:#f8fafc;}
td.hf-msg{min-width:260px; color:#33414f;}
table.hf-table td:nth-child(1), table.hf-table td:nth-child(2){white-space:nowrap;}

/* KPI + bars */
.hf-kpis{display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:6px 0 16px;}
.hf-kpi{background:#fff; border:1px solid var(--hf-line); border-top:4px solid var(--hf-navy-2); border-radius:6px; padding:12px 14px;}
.hf-kpi.crit{border-top-color:var(--hf-crit);} .hf-kpi.high{border-top-color:var(--hf-high);}
.hf-kpi.norm{border-top-color:var(--hf-norm);}
.hf-kpi .k{font-size:.72rem; text-transform:uppercase; letter-spacing:.05em; color:var(--hf-muted);}
.hf-kpi .v{font-size:1.9rem; font-weight:700; color:var(--hf-navy); line-height:1.15;}
.hf-kpi .s{font-size:.78rem; color:var(--hf-muted);}
.hf-panel{background:#fff; border:1px solid var(--hf-line); border-radius:6px; padding:14px 16px; margin-bottom:12px;}
.hf-panel h4{margin:0 0 10px; font-size:.95rem; color:var(--hf-navy);}
.hf-bar-row{display:grid; grid-template-columns:150px 1fr 40px; gap:10px; align-items:center; margin:7px 0; font-size:.85rem;}
.hf-bar-track{background:#eceff2; border-radius:3px; height:12px; overflow:hidden;}
.hf-bar-fill{height:100%; border-radius:3px;}
.hf-bar-n{text-align:right; font-weight:600; color:var(--hf-ink);}
.hf-empty{border:1px dashed #b9c3cc; border-radius:6px; padding:28px; text-align:center; color:var(--hf-muted); background:#fff;}
.hf-footer{margin-top:28px; padding-top:12px; border-top:1px solid var(--hf-line); font-size:.78rem; color:var(--hf-muted);}

/* native widgets */
button[kind="primary"]{background:var(--hf-navy) !important; border-color:var(--hf-navy) !important;}
button[kind="primary"], button[kind="primary"] *{color:#ffffff !important;}
button[kind="primary"]:hover{background:var(--hf-navy-2) !important;}
button[kind="primary"]:disabled{background:#b8c3cf !important; border-color:#b8c3cf !important; opacity:1;}
div[role="radiogroup"]{gap:8px; padding-bottom:12px; margin-bottom:6px; border-bottom:1px solid var(--hf-line);}
[data-testid="stRadioOption"], label[data-baseweb="radio"]{background:#fff; border:1px solid var(--hf-line);
  border-radius:4px; padding:6px 16px; margin:0; cursor:pointer;}
[data-testid="stRadioOption"] > div > div:first-child, label[data-baseweb="radio"] > div:first-child{display:none;}
[data-testid="stRadioOption"] p, label[data-baseweb="radio"] p{font-weight:600; color:var(--hf-navy); margin:0;}
[data-testid="stRadioOption"][data-selected="true"], label[data-baseweb="radio"]:has(input:checked){
  background:var(--hf-navy); border-color:var(--hf-navy);}
[data-testid="stRadioOption"][data-selected="true"] p, label[data-baseweb="radio"]:has(input:checked) p{color:#fff;}
</style>
"""


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
    css = "openai" if mode == "OpenAI" else "local"
    return f'<span class="hf-chip {css}">{esc(mode)}</span>'


def provisional_chip(case: dict) -> str:
    return '<span class="hf-chip prov">Provisional</span>' if case.get("provisional") else ""


def needs_review(case: dict) -> bool:
    return bool(case.get("escalate_to_human")) and not case.get("human_reviewed")


def review_chip(case: dict) -> str:
    if case.get("human_reviewed"):
        return '<span class="hf-chip done">Human reviewed</span>'
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
        ("Confidence", confidence_text(case)),
    ]
    facts_html = "".join(
        f'<div class="hf-fact"><div class="k">{k}</div><div class="v">{esc(v)}</div></div>' for k, v in facts
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
    if case.get("analysis_mode") == "OpenAI":
        notes += (
            '<div class="hf-note info">Confidence is not shown for OpenAI results because the model does '
            "not provide a calibrated probability.</div>"
        )

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
        "<th>Urgency</th><th>Category</th><th>Routed to</th><th>Source</th><th>Review</th><th>Message</th>"
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


# ---------------------------------------------------------------------------
# Human review
# ---------------------------------------------------------------------------

def render_review(case: dict, prefix: str):
    case_id = case["id"]
    st.markdown("### Human review")

    def review_form(default: str, default_notes: str, button_label: str):
        key = f"{prefix}_{case_id}"
        final = st.selectbox(
            "Final urgency",
            URGENCY_ORDER,
            index=URGENCY_ORDER.index(default) if default in URGENCY_ORDER else 2,
            key=f"final_{key}",
            help="Defaults to the AI recommendation. Change it to override.",
        )
        notes = st.text_input("Reviewer notes (optional)", value=default_notes, key=f"notes_{key}")
        with st.expander("Draft acknowledgement (not sent automatically)"):
            st.text_area("Draft", value=ACK_TEMPLATES[final], height=110, key=f"ack_{key}_{final}")
            st.caption("A draft only. Staff decide whether and how to use it.")
        if st.button(button_label, type="primary", key=f"save_{key}"):
            try:
                updated = api(
                    "POST", f"/cases/{case_id}/review", timeout=15,
                    json={"human_urgency": final, "human_notes": notes},
                )
                st.session_state["flash"] = f"Decision saved for case #{updated['id']}."
                if st.session_state.get("last_result", {}) and st.session_state["last_result"].get("id") == case_id:
                    st.session_state["last_result"] = updated
                st.rerun()
            except requests.exceptions.RequestException as exc:
                st.error(explain_error(exc))

    if case.get("human_reviewed"):
        outcome = "Overridden by reviewer" if was_overridden(case) else "Confirmed by reviewer"
        note = f' &middot; &ldquo;{esc(case["human_notes"])}&rdquo;' if case.get("human_notes") else ""
        st.markdown(
            f'<div class="hf-note ok"><b>{outcome}.</b> AI recommended {esc(case.get("urgency"))}; '
            f'final urgency is <b>{esc(case.get("human_urgency"))}</b>{note}'
            f' &middot; {esc(fmt_time(case.get("reviewed_at")))}</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Change this decision"):
            review_form(case.get("human_urgency"), case.get("human_notes") or "", "Update decision")
    else:
        if case.get("escalate_to_human"):
            st.caption("Review required. Confirm the AI recommendation or override it.")
        else:
            st.caption("Review is optional for this case. You can still confirm or override it.")
        review_form(case.get("urgency", "Normal"), "", "Save decision")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HumanFirst AI",
    page_icon="\U0001F4E8",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(CSS, unsafe_allow_html=True)

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
    st.subheader("Connection")
    st.session_state["backend_url"] = st.text_input("Backend URL", value=st.session_state["backend_url"])
    st.caption("Cases are stored by the backend and persist across restarts.")

health = get_health()
render_header(health)

if st.session_state["flash"]:
    st.success(st.session_state["flash"])
    st.session_state["flash"] = None

cases, cases_error = get_cases()
if health is None:
    st.error(
        f"The backend at {st.session_state['backend_url']} is not reachable. "
        "Start it with `python app.py` in the backend folder, or change the URL in the sidebar."
    )

# A radio (not st.tabs) so the chosen section survives the reruns that follow saving a decision.
PAGES = ["Triage a message", "Review queue", "Batch upload", "Dashboard"]
page = st.radio("Section", PAGES, horizontal=True, label_visibility="collapsed", key="page")

# ---- Triage ---------------------------------------------------------------
if page == PAGES[0]:
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

        f1, f2, f3 = st.columns(3)
        status = f1.selectbox("Status", ["Awaiting review", "Reviewed", "All cases"])
        levels = f2.multiselect("Urgency", URGENCY_ORDER, default=URGENCY_ORDER)
        modes = f3.multiselect("Source", ["OpenAI", "Local ML"], default=["OpenAI", "Local ML"])

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
            chosen = st.selectbox(
                "Case", options, key="queue_case",
                format_func=lambda i: "Select a case..." if i == 0
                else f"#{i} · {current_urgency(by_id[i])} · {(by_id[i].get('message') or '')[:70]}",
            )
            if chosen in by_id:
                render_case(by_id[chosen])
                render_review(by_id[chosen], "queue")

# ---- Batch -----------------------------------------------------------------
if page == PAGES[2]:
    st.markdown("### Batch upload")
    st.write(
        "Upload a CSV with a `message` column. Each row goes through the same pipeline as a single "
        "message and is saved to the case store."
    )
    st.caption("Each row is sent to OpenAI individually, so large files can take a few minutes.")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded and st.button("Analyse batch", type="primary"):
        with st.spinner(f"Analysing {uploaded.name}..."):
            try:
                st.session_state["last_batch"] = api(
                    "POST", "/analyze", timeout=900,
                    files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
                )
                st.session_state["flash"] = f"Analysed and saved {st.session_state['last_batch']['summary']['total_messages']} messages."
                st.rerun()
            except requests.exceptions.RequestException as exc:
                st.error(explain_error(exc))

    batch = st.session_state["last_batch"]
    if batch:
        results = batch.get("results", [])
        summary = batch.get("summary", {})
        st.markdown("### Last batch")
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
                ("OpenAI", sum(c.get("analysis_mode") == "OpenAI" for c in cases), "#1d466f"),
                ("Local ML (provisional)", len(provisional), "#8a97a3"),
            ])
        with c2:
            render_bars("Category", [(cat, sum(c.get("category") == cat for c in cases), "#1d466f") for cat in CATEGORIES])
            render_bars("Review status", [
                ("Awaiting review", sum(needs_review(c) for c in cases), "#c77700"),
                ("Reviewed", len(reviewed), "#3f7d5a"),
                ("No review required", sum(not c.get("escalate_to_human") and not c.get("human_reviewed") for c in cases), "#8a97a3"),
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
