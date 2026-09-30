import io
from datetime import datetime

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Backend wiring
# ---------------------------------------------------------------------------
# The backend now PERSISTS every analyzed case to a database (see the
# backend's database.py) instead of only living in this app's session
# memory. That means:
#   - Past cases survive a Streamlit restart, a page refresh, or a
#     different browser session entirely.
#   - /triage and /analyze both return the STORED case (it has an "id"),
#     and a human review is attached to that exact case via
#     POST /cases/<id>/review rather than just kept in this tab's memory.
#   - Priority Queue / Dashboard read from GET /cases each time they're
#     shown, so they always reflect everything ever analyzed, not just
#     what happened in the current browser tab.

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
    """Analyzes AND persists the message. The returned dict includes 'id'."""
    response = requests.post(f"{base_url}/triage", json={"message": message}, timeout=30)
    response.raise_for_status()
    return response.json()


def analyse_batch(base_url: str, uploaded_file) -> dict:
    """Analyzes AND persists every row. Returns {"summary": ..., "results": [...]}."""
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
    response = requests.post(f"{base_url}/analyze", files=files, timeout=60)
    response.raise_for_status()
    return response.json()


def fetch_cases(base_url: str, **filters) -> list[dict]:
    """Fetch stored cases from the backend — this is the persisted history,
    not just what happened in this browser session."""
    params = {k: v for k, v in filters.items() if v is not None}
    response = requests.get(f"{base_url}/cases", params=params, timeout=15)
    response.raise_for_status()
    return response.json()["cases"]


def submit_review(base_url: str, case_id: int, human_urgency: str, human_notes: str) -> dict:
    """Attach a human reviewer's decision to a specific stored case."""
    response = requests.post(
        f"{base_url}/cases/{case_id}/review",
        json={"human_urgency": human_urgency, "human_notes": human_notes},
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def delete_all_cases(base_url: str) -> dict:
    response = requests.delete(f"{base_url}/cases", timeout=15)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short_time(iso_timestamp: str) -> str:
    try:
        return datetime.fromisoformat(iso_timestamp).strftime("%H:%M:%S")
    except (TypeError, ValueError):
        return ""


def cases_to_csv(cases: list[dict]) -> str:
    if not cases:
        return ""
    import csv as csv_module

    fieldnames = [
        "id", "created_at", "message", "urgency", "urgency_confidence", "category",
        "category_confidence", "route", "escalation", "escalate_to_human",
        "attention_language_detected", "source", "human_reviewed", "human_urgency",
        "human_notes", "reviewed_at",
    ]
    buffer = io.StringIO()
    writer = csv_module.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in cases:
        writer.writerow(row)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def init_state():
    if "backend_url" not in st.session_state:
        st.session_state["backend_url"] = DEFAULT_BACKEND_URL
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None  # AI result awaiting human review
    if "last_error" not in st.session_state:
        st.session_state["last_error"] = None
    if "last_batch_summary" not in st.session_state:
        st.session_state["last_batch_summary"] = None
    if "confirm_delete_all" not in st.session_state:
        st.session_state["confirm_delete_all"] = False


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
    st.caption("Past cases are stored on the backend, so they persist across restarts and sessions.")

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
        st.caption(f"Case #{result['id']} — saved to the case store")
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

        if result.get("human_reviewed"):
            st.success(
                f"Already reviewed: final urgency **{result['human_urgency']}**"
                + (f" — \"{result['human_notes']}\"" if result.get("human_notes") else "")
            )
        else:
            st.caption("Confirm or override the AI's call — this is saved to the case permanently.")

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

            if st.button("Confirm & save this decision", type="primary"):
                try:
                    updated = submit_review(
                        st.session_state["backend_url"], result["id"], human_urgency, human_notes
                    )
                    st.session_state["last_result"] = updated
                    st.success(f"Saved to case #{updated['id']}.")
                    st.rerun()
                except requests.exceptions.RequestException as e:
                    st.error(f"Couldn't save the review: {e}")

    elif not st.session_state["last_error"]:
        st.write("Enter a message above and click **Analyse** to see the result.")

# ---------------------------------------------------------------------------
# Priority Queue tab — every stored case, most urgent first
# ---------------------------------------------------------------------------
with tab_queue:
    st.caption(
        "Every case ever analyzed by this backend — sorted most urgent first, then by "
        "whether it's flagged for human review. This is the order a caseworker would "
        "want to work through the inbox, and it's the same across restarts and sessions."
    )

    try:
        cases = fetch_cases(st.session_state["backend_url"])
    except requests.exceptions.RequestException:
        cases = None
        st.error(f"Couldn't reach the backend at {st.session_state['backend_url']}.")

    if cases is not None:
        if not cases:
            st.write("No cases stored yet. Analyse a message or upload a batch CSV.")
        else:
            queue = sorted(
                cases,
                key=lambda c: (
                    URGENCY_RANK.get(c["urgency"], 99),
                    0 if c["escalate_to_human"] else 1,
                    -c["urgency_confidence"],
                ),
            )

            rows = []
            for item in queue:
                urgency_badge = URGENCY_COLORS.get(item["urgency"], "⚪")
                rows.append({
                    "": urgency_badge,
                    "ID": item["id"],
                    "Urgency": item["urgency"],
                    "Category": item["category"],
                    "Route": item["route"],
                    "Escalate?": "Yes" if item["escalate_to_human"] else "No",
                    "Confidence": f"{item['urgency_confidence']:.0%}",
                    "Reviewed?": "Yes" if item["human_reviewed"] else "No",
                    "Message": item["message"][:80] + ("..." if len(item["message"]) > 80 else ""),
                    "Time": _short_time(item["created_at"]),
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Batch Upload tab — CSV of many messages at once
# ---------------------------------------------------------------------------
with tab_batch:
    st.write(
        "Upload a CSV with a `message` column (any incoming messages, no labels needed) "
        "to triage them all at once. Every row is saved to the case store immediately — "
        "check the Priority Queue or Dashboard tabs to see them."
    )
    uploaded_file = st.file_uploader("CSV file", type=["csv"])

    if uploaded_file and st.button("Analyse batch", type="primary"):
        with st.spinner(f"Analysing {uploaded_file.name}..."):
            try:
                batch_result = analyse_batch(st.session_state["backend_url"], uploaded_file)
                st.session_state["last_batch_summary"] = batch_result["summary"]
                st.success(f"Analysed and saved {batch_result['summary']['total_messages']} messages.")
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
    try:
        cases = fetch_cases(st.session_state["backend_url"])
    except requests.exceptions.RequestException:
        cases = None
        st.error(f"Couldn't reach the backend at {st.session_state['backend_url']}.")

    if cases is not None:
        if not cases:
            st.write("No cases stored yet. Analyse one, or upload a batch, to populate the dashboard.")
        else:
            total = len(cases)
            critical = sum(1 for c in cases if c["urgency"] == "Critical")
            high = sum(1 for c in cases if c["urgency"] == "High")
            normal = sum(1 for c in cases if c["urgency"] == "Normal")
            escalated = sum(1 for c in cases if c["escalate_to_human"])
            escalation_rate = (escalated / total * 100) if total else 0
            avg_confidence = sum(c["urgency_confidence"] for c in cases) / total
            reviewed = sum(1 for c in cases if c["human_reviewed"])
            overridden = sum(
                1 for c in cases
                if c["human_reviewed"] and c["human_urgency"] != c["urgency"]
            )

            st.subheader("Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total analysed (all-time)", total)
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
            for c in cases:
                if c.get("category") in category_counts:
                    category_counts[c["category"]] += 1
            st.bar_chart(category_counts)

            st.subheader("Recent cases")
            for item in cases[:10]:  # already most-recent-first from the backend
                urgency_badge = URGENCY_COLORS.get(item["urgency"], "⚪")
                dept_icon = DEPARTMENT_ICONS.get(item["category"], "📁")
                escalate_flag = " · escalated" if item["escalate_to_human"] else ""
                override_flag = (
                    " · overridden by human"
                    if item["human_reviewed"] and item["human_urgency"] != item["urgency"]
                    else ""
                )
                st.write(
                    f"#{item['id']} {urgency_badge} **{item['urgency']}** · {dept_icon} {item['category']}"
                    f"{escalate_flag}{override_flag} · {item['urgency_confidence']:.0%} confidence "
                    f"· {_short_time(item['created_at'])}"
                )

            st.divider()
            col_export, col_clear = st.columns(2)
            with col_export:
                csv_data = cases_to_csv(cases)
                st.download_button(
                    "Export all cases as CSV",
                    data=csv_data,
                    file_name=f"humanfirst_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
            with col_clear:
                st.session_state["confirm_delete_all"] = st.checkbox(
                    "I understand this permanently deletes ALL stored cases",
                    value=st.session_state["confirm_delete_all"],
                )
                if st.button(
                    "Delete all stored cases",
                    disabled=not st.session_state["confirm_delete_all"],
                ):
                    try:
                        result = delete_all_cases(st.session_state["backend_url"])
                        st.session_state["confirm_delete_all"] = False
                        st.session_state["last_batch_summary"] = None
                        st.success(f"Deleted {result['deleted']} cases.")
                        st.rerun()
                    except requests.exceptions.RequestException as e:
                        st.error(f"Couldn't delete cases: {e}")