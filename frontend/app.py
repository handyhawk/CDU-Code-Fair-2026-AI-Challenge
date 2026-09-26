import csv
import io
import uuid
from datetime import datetime

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Backend wiring
# ---------------------------------------------------------------------------

DEFAULT_BACKEND_URL = "http://localhost:5000"

URGENCY_COLORS = {"Critical": "🔴", "High": "🟠", "Normal": "🟢"}
URGENCY_RANK = {"Critical": 0, "High": 1, "Normal": 2}

DEPARTMENT_ICONS = {
    "Health & Safety": "🏥",
    "Housing & Utilities": "🏠",
    "Financial Support": "💰",
    "Licensing & Services": "📋",
    "General Enquiries": "✉️",
}

# Simple canned acknowledgement templates, client-side only — no AI call.
# These are drafts for a human to review/edit before sending, never sent
# automatically. Keeps the "AI assists, human decides" principle visible
# even in this small UI touch.
ACK_TEMPLATES = {
    "Critical": (
        "Thank you for reaching out. We understand this is urgent — a team "
        "member will contact you within the next few hours. If you are in "
        "immediate danger, please call 000."
    ),
    "High": (
        "Thank you for your message. We've noted the time-sensitive nature "
        "of your request and a caseworker will follow up within 1–2 "
        "business days."
    ),
    "Normal": (
        "Thank you for contacting us. We've received your message and will "
        "respond as soon as possible, typically within 5 business days."
    ),
}


# ---------------------------------------------------------------------------
# Backend calls
# ---------------------------------------------------------------------------

def check_backend_health(base_url: str) -> dict | None:
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


def analyse_message(base_url: str, message: str) -> dict:
    payload = {"message": message}
    response = requests.post(f"{base_url}/triage", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def analyse_batch(base_url: str, uploaded_file) -> dict:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
    response = requests.post(f"{base_url}/analyze", files=files, timeout=60)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def init_state():
    if "backend_url" not in st.session_state:
        st.session_state["backend_url"] = DEFAULT_BACKEND_URL
    if "history" not in st.session_state:
        st.session_state["history"] = []  # list of dicts: AI result + review fields + timestamp
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "last_error" not in st.session_state:
        st.session_state["last_error"] = None
    if "last_batch_summary" not in st.session_state:
        st.session_state["last_batch_summary"] = None


def log_entry(result: dict, human_urgency: str | None, human_notes: str, reviewed: bool):
    """Add one analysed message (single or batch) to the shared history."""
    was_overridden = bool(reviewed and human_urgency and human_urgency != result["urgency"])
    st.session_state["history"].append({
        **result,
        "analysed_at": datetime.now(),
        "human_reviewed": reviewed,
        "human_urgency": human_urgency,
        "human_notes": human_notes,
        "was_overridden": was_overridden,
    })


def history_to_csv(history: list[dict]) -> str:
    if not history:
        return ""
    fieldnames = [
        "analysed_at", "message", "urgency", "urgency_confidence", "category",
        "category_confidence", "route", "escalation", "escalate_to_human",
        "attention_language_detected", "human_reviewed", "human_urgency",
        "was_overridden", "human_notes",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in history:
        row_copy = dict(row)
        row_copy["analysed_at"] = row["analysed_at"].strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow(row_copy)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="HumanFirst AI", page_icon="📨", layout="centered")
init_state()

with st.sidebar:
    st.subheader("Backend connection")
    st.session_state["backend_url"] = st.text_input(
        "Backend URL", value=st.session_state["backend_url"]
    )
    health = check_backend_health(st.session_state["backend_url"])
    if health is None:
        st.error("⚫ Backend unreachable")
    elif health.get("model_trained"):
        st.success("🟢 Backend connected — model trained")
    else:
        st.warning("🟡 Backend connected — model NOT trained yet (call /train)")

st.title("📨 HumanFirst AI")
st.caption("AI-assisted government inbox: urgency detection, routing, and escalation")

tab_analyse, tab_queue, tab_batch, tab_dashboard = st.tabs(
    ["Analyse", "Priority Queue", "Batch Upload", "Dashboard"]
)

# ---------------------------------------------------------------------------
# Analyse tab — single message + human review controls
# ---------------------------------------------------------------------------
with tab_analyse:
    message = st.text_area(
        "Incoming message",
        height=150,
        placeholder="Paste or type the message to analyse...",
        key="message_input",
    )

    analyse_clicked = st.button("Analyse", type="primary", disabled=not message.strip())

    st.divider()

    if analyse_clicked:
        with st.spinner("Analysing..."):
            try:
                result = analyse_message(st.session_state["backend_url"], message)
                st.session_state["last_result"] = result
                st.session_state["last_error"] = None
            except requests.exceptions.RequestException:
                st.session_state["last_result"] = None
                st.session_state["last_error"] = (
                    f"Couldn't reach the backend at {st.session_state['backend_url']}. "
                    "Make sure app.py (the backend) is running."
                )

    if st.session_state["last_error"]:
        st.error(st.session_state["last_error"])

    if st.session_state["last_result"]:
        result = st.session_state["last_result"]

        urgency_badge = URGENCY_COLORS.get(result["urgency"], "⚪")
        dept_icon = DEPARTMENT_ICONS.get(result["category"], "📁")

        st.subheader("AI assessment")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Urgency", f"{urgency_badge} {result['urgency']}")
        with col2:
            st.metric("Category", f"{dept_icon} {result['category']}")
        with col3:
            st.metric("Confidence", f"{result['urgency_confidence']:.0%}")

        col4, col5, col6 = st.columns(3)
        with col4:
            st.write(f"**Route to:** {result['route']}")
        with col5:
            st.write(f"**Escalation tier:** {result['escalation']}")
        with col6:
            st.write(f"**Flag for human:** {'Yes' if result['escalate_to_human'] else 'No'}")

        st.write("**Explanation:**")
        st.info(result["explanation"])

        if result.get("matched_risk_keywords"):
            st.write("**Risk factors identified:**")
            st.warning(", ".join(result["matched_risk_keywords"]))

        if result.get("attention_language_detected"):
            st.caption(
                "⚠️ Uses urgent-sounding language — this did not by itself raise "
                "the urgency score. Verify against the risk factors above."
            )

        # ---- Human review controls (the AI suggests, a human decides) ----
        st.divider()
        st.subheader("Human review")
        st.caption("Confirm or override the AI's call before it's logged.")

        urgency_options = ["Critical", "High", "Normal"]
        human_urgency = st.selectbox(
            "Final urgency (defaults to the AI's call)",
            options=urgency_options,
            index=urgency_options.index(result["urgency"]),
            key="human_urgency_select",
        )
        human_notes = st.text_input("Reviewer notes (optional)", key="human_notes_input")

        draft = ACK_TEMPLATES.get(human_urgency, ACK_TEMPLATES["Normal"])
        with st.expander("Draft acknowledgement (AI-suggested, edit before sending)"):
            st.text_area("Draft", value=draft, height=100, key="draft_ack_text")
            st.caption("This is not sent automatically — copy it into your own system if suitable.")

        if st.button("Confirm & log this decision", type="primary"):
            log_entry(result, human_urgency, human_notes, reviewed=True)
            st.session_state["last_result"] = None
            st.success("Logged. Ready for the next message.")
            st.rerun()

    elif not st.session_state["last_error"]:
        st.write("Enter a message above and click **Analyse** to see the result.")

# ---------------------------------------------------------------------------
# Priority Queue tab — everything analysed so far, most urgent first
# ---------------------------------------------------------------------------
with tab_queue:
    history = st.session_state["history"]

    if not history:
        st.write("No messages in the queue yet. Analyse a message or upload a batch CSV.")
    else:
        st.caption(
            "Sorted most urgent first, then by whether it's flagged for human review — "
            "this is the order a caseworker would want to work through the inbox."
        )

        queue = sorted(
            history,
            key=lambda h: (
                URGENCY_RANK.get(h["urgency"], 99),
                0 if h["escalate_to_human"] else 1,
                -h["urgency_confidence"],
            ),
        )

        rows = []
        for item in queue:
            urgency_badge = URGENCY_COLORS.get(item["urgency"], "⚪")
            rows.append({
                "": urgency_badge,
                "Urgency": item["urgency"],
                "Category": item["category"],
                "Route": item["route"],
                "Escalate?": "Yes" if item["escalate_to_human"] else "No",
                "Confidence": f"{item['urgency_confidence']:.0%}",
                "Reviewed?": "Yes" if item["human_reviewed"] else "No",
                "Message": item["message"][:80] + ("..." if len(item["message"]) > 80 else ""),
                "Time": item["analysed_at"].strftime("%H:%M:%S"),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Batch Upload tab — CSV of many messages at once
# ---------------------------------------------------------------------------
with tab_batch:
    st.write(
        "Upload a CSV with a `message` column (any incoming messages, no labels needed) "
        "to triage them all at once. Results are added to the Priority Queue and Dashboard."
    )
    uploaded_file = st.file_uploader("CSV file", type=["csv"])

    if uploaded_file and st.button("Analyse batch", type="primary"):
        with st.spinner(f"Analysing {uploaded_file.name}..."):
            try:
                batch_result = analyse_batch(st.session_state["backend_url"], uploaded_file)
                st.session_state["last_batch_summary"] = batch_result["summary"]
                for row_result in batch_result["results"]:
                    log_entry(row_result, human_urgency=None, human_notes="", reviewed=False)
                st.success(f"Analysed {batch_result['summary']['total_messages']} messages.")
            except requests.exceptions.RequestException as e:
                st.error(f"Batch analysis failed: {e}")

    if st.session_state["last_batch_summary"]:
        st.subheader("Last batch summary")
        summary = st.session_state["last_batch_summary"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Total messages", summary["total_messages"])
        col2.metric("Escalated", f"{summary['escalated_count']} ({summary['escalation_rate']:.0%})")
        col3.metric("Avg. confidence", f"{summary['average_urgency_confidence']:.0%}")
        st.write("**By urgency:**", summary["by_urgency"])
        st.write("**By category:**", summary["by_category"])

# ---------------------------------------------------------------------------
# Dashboard tab
# ---------------------------------------------------------------------------
with tab_dashboard:
    history = st.session_state["history"]

    if not history:
        st.write("No messages analysed yet this session. Analyse one, or upload a batch, to populate the dashboard.")
    else:
        total = len(history)
        critical = sum(1 for h in history if h["urgency"] == "Critical")
        high = sum(1 for h in history if h["urgency"] == "High")
        normal = sum(1 for h in history if h["urgency"] == "Normal")
        escalated = sum(1 for h in history if h["escalate_to_human"])
        escalation_rate = (escalated / total * 100) if total else 0
        avg_confidence = sum(h["urgency_confidence"] for h in history) / total
        overridden = sum(1 for h in history if h["was_overridden"])
        reviewed = sum(1 for h in history if h["human_reviewed"])

        st.subheader("Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total analysed", total)
        col2.metric("Escalated to human", f"{escalated} ({escalation_rate:.0f}%)")
        col3.metric("Avg. confidence", f"{avg_confidence:.0%}")

        col4, col5 = st.columns(2)
        col4.metric("Human-reviewed", reviewed)
        col5.metric("Human overrode AI", overridden)

        st.subheader("Urgency breakdown")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("🔴 Critical", critical)
        col_b.metric("🟠 High", high)
        col_c.metric("🟢 Normal", normal)
        st.bar_chart({"Critical": critical, "High": high, "Normal": normal})

        st.subheader("Category breakdown")
        category_counts = {dept: 0 for dept in DEPARTMENT_ICONS}
        for h in history:
            category_counts[h["category"]] = category_counts.get(h["category"], 0) + 1
        st.bar_chart(category_counts)

        st.subheader("Recent messages")
        recent = list(reversed(history[-10:]))  # most recent first
        for item in recent:
            urgency_badge = URGENCY_COLORS.get(item["urgency"], "⚪")
            dept_icon = DEPARTMENT_ICONS.get(item["category"], "📁")
            timestamp = item["analysed_at"].strftime("%H:%M:%S")
            escalate_flag = " · escalated" if item["escalate_to_human"] else ""
            override_flag = " · overridden by human" if item["was_overridden"] else ""
            st.write(
                f"{urgency_badge} **{item['urgency']}** · {dept_icon} {item['category']}"
                f"{escalate_flag}{override_flag} · {item['urgency_confidence']:.0%} confidence · {timestamp}"
            )

        st.divider()
        col_clear, col_export = st.columns(2)
        with col_clear:
            if st.button("Clear dashboard history"):
                st.session_state["history"] = []
                st.session_state["last_batch_summary"] = None
                st.rerun()
        with col_export:
            csv_data = history_to_csv(history)
            st.download_button(
                "Export history as CSV",
                data=csv_data,
                file_name=f"humanfirst_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )