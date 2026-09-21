"""
Run the backend first, then run this with:
    streamlit run app.py
"""
import uuid
import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000/triage"


def analyse_message(message: str) -> dict:
    """Calls the backend's /triage endpoint. Raises on network/HTTP errors."""
    payload = {
        "message_id": str(uuid.uuid4()),
        "content": message,
    }
    response = requests.post(BACKEND_URL, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="HumanFirst AI - Inbox Assistant", layout="centered")

st.title("HumanFirst AI")
st.caption("AI-assisted government inbox: urgency detection, routing, and escalation")

message = st.text_area(
    "Incoming message",
    height=150,
    placeholder="Paste or type the message to analyse...",
)

analyse_clicked = st.button("Analyse", type="primary", disabled=not message.strip())

st.divider()

if analyse_clicked:
    with st.spinner("Analysing..."):
        try:
            result = analyse_message(message)
            st.session_state["last_result"] = result
            st.session_state["last_error"] = None
        except requests.exceptions.RequestException as e:
            st.session_state["last_error"] = (
                "Couldn't reach the backend. Make sure it's running at "
                "http://localhost:8000 (see backend/run.py)."
            )

if st.session_state.get("last_error"):
    st.error(st.session_state["last_error"])

if "last_result" in st.session_state and st.session_state.get("last_result"):
    result = st.session_state["last_result"]

    urgency_colors = {"Critical": "🔴", "High": "🟠", "Normal": "🟢"}
    badge = urgency_colors.get(result["final_urgency"], "⚪")

    st.subheader("Result")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Urgency", f"{badge} {result['final_urgency']}")
        st.write(f"**Route to:** {result['final_route']}")

    with col2:
        st.write(f"**Human escalation:** {'Yes' if result['escalate_to_human'] else 'No'}")
        st.write(f"**Message ID:** `{result['message_id']}`")

    st.write("**Explanation:**")
    st.info(result["explanation"])

    if result.get("safety_triggers"):
        st.write("**Safety triggers detected:**")
        for trigger in result["safety_triggers"]:
            st.warning(trigger)

    if result.get("draft_acknowledgement"):
        st.write("**Draft acknowledgement (auto-suggested, not sent):**")
        st.code(result["draft_acknowledgement"])
else:
    if not st.session_state.get("last_error"):
        st.write("Enter a message above and click **Analyse** to see the result.")