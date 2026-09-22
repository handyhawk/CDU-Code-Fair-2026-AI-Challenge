import uuid
from datetime import datetime

import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000/triage"

URGENCY_COLORS = {"Critical": "🔴", "High": "🟠", "Normal": "🟢"}

DEPARTMENT_ICONS = {
    "Health & Safety": "🏥",
    "Housing & Utilities": "🏠",
    "Financial Support": "💰",
    "Licensing & Services": "📋",
    "General Enquiries": "✉️",
}


def analyse_message(message: str) -> dict:
    payload = {
        "message_id": str(uuid.uuid4()),
        "content": message,
    }
    response = requests.post(BACKEND_URL, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def init_state():
    if "history" not in st.session_state:
        st.session_state["history"] = []  # list of dicts: result + timestamp
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "last_error" not in st.session_state:
        st.session_state["last_error"] = None


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="HumanFirst AI", page_icon="📨", layout="centered")
init_state()

st.title("📨 HumanFirst AI")
st.caption("AI-assisted government inbox: urgency detection, routing, and escalation")

tab_analyse, tab_dashboard = st.tabs(["Analyse", "Dashboard"])

# ---------------------------------------------------------------------------
# Analyse tab
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
                result = analyse_message(message)
                st.session_state["last_result"] = result
                st.session_state["last_error"] = None
                st.session_state["history"].append(
                    {**result, "analysed_at": datetime.now()}
                )
            except requests.exceptions.RequestException:
                st.session_state["last_error"] = (
                    "Couldn't reach the backend. Make sure it's running at "
                    "http://localhost:8000."
                )

    if st.session_state["last_error"]:
        st.error(st.session_state["last_error"])

    if st.session_state["last_result"]:
        result = st.session_state["last_result"]

        urgency_badge = URGENCY_COLORS.get(result["final_urgency"], "⚪")
        dept_icon = DEPARTMENT_ICONS.get(result["category"], "📁")

        st.subheader("Result")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Urgency", f"{urgency_badge} {result['final_urgency']}")

        with col2:
            st.metric("Category", f"{dept_icon} {result['category']}")

        with col3:
            st.metric("Confidence", f"{result['confidence_score']:.0%}")

        col4, col5 = st.columns(2)
        with col4:
            st.write(f"**Route to:** {result['final_route']}")
        with col5:
            st.write(f"**Human escalation:** {'Yes' if result['escalate_to_human'] else 'No'}")

        st.write("**Explanation:**")
        st.info(result["explanation"])

        if result.get("safety_triggers"):
            st.write("**Safety triggers detected:**")
            for trigger in result["safety_triggers"]:
                st.warning(trigger)

        if result.get("draft_acknowledgement"):
            st.write("**Draft acknowledgement (auto-suggested, not sent):**")
            st.code(result["draft_acknowledgement"])
    elif not st.session_state["last_error"]:
        st.write("Enter a message above and click **Analyse** to see the result.")

# ---------------------------------------------------------------------------
# Dashboard tab
# ---------------------------------------------------------------------------
with tab_dashboard:
    history = st.session_state["history"]

    if not history:
        st.write("No messages analysed yet this session. Analyse one to populate the dashboard.")
    else:
        total = len(history)
        critical = sum(1 for h in history if h["final_urgency"] == "Critical")
        high = sum(1 for h in history if h["final_urgency"] == "High")
        normal = sum(1 for h in history if h["final_urgency"] == "Normal")
        escalated = sum(1 for h in history if h["escalate_to_human"])
        escalation_rate = (escalated / total * 100) if total else 0
        avg_confidence = sum(h["confidence_score"] for h in history) / total

        st.subheader("Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total analysed", total)
        col2.metric("Escalated to human", f"{escalated} ({escalation_rate:.0f}%)")
        col3.metric("Avg. confidence", f"{avg_confidence:.0%}")

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
            urgency_badge = URGENCY_COLORS.get(item["final_urgency"], "⚪")
            dept_icon = DEPARTMENT_ICONS.get(item["category"], "📁")
            timestamp = item["analysed_at"].strftime("%H:%M:%S")
            escalate_flag = " · escalated" if item["escalate_to_human"] else ""
            st.write(
                f"{urgency_badge} **{item['final_urgency']}** · {dept_icon} {item['category']}"
                f"{escalate_flag} · {item['confidence_score']:.0%} confidence · {timestamp}"
            )

        if st.button("Clear dashboard history"):
            st.session_state["history"] = []
            st.rerun()