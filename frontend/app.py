import csv as csv_module
import io
from datetime import datetime

import requests
import streamlit as st


# ---------------------------------------------------------------------------
# Backend wiring
# ---------------------------------------------------------------------------

DEFAULT_BACKEND_URL = "http://localhost:5000"


URGENCY_COLORS = {
    "Critical": "🔴",
    "High": "🟠",
    "Normal": "🟢",
}


URGENCY_RANK = {
    "Critical": 0,
    "High": 1,
    "Normal": 2,
}


DEPARTMENT_ICONS = {
    "Health & Safety": "🏥",
    "Housing & Utilities": "🏠",
    "Financial Support": "💰",
    "Licensing & Services": "📋",
    "General Enquiries": "✉️",
}


# ---------------------------------------------------------------------------
# Draft acknowledgements
# ---------------------------------------------------------------------------

# These are drafts only.
# They are never sent automatically.
ACK_TEMPLATES = {
    "Critical": (
        "Thank you for contacting us. Your message has been identified "
        "as requiring urgent human review. A staff member will assess "
        "the information provided as soon as possible."
    ),

    "High": (
        "Thank you for your message. Your request has been marked for "
        "priority human review."
    ),

    "Normal": (
        "Thank you for contacting us. Your message has been received "
        "and will be reviewed through the standard process."
    ),
}


# ---------------------------------------------------------------------------
# Backend API calls
# ---------------------------------------------------------------------------

def check_backend_health(
    base_url: str,
) -> dict | None:
    """
    Check whether the Flask backend is reachable.
    """

    try:
        response = requests.get(
            f"{base_url}/health",
            timeout=5,
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException:
        return None


def analyse_message(
    base_url: str,
    message: str,
) -> dict:
    """
    Analyse one incoming message.

    The backend:
    - chooses OpenAI or Local ML fallback
    - persists the result
    - returns the stored case
    """

    response = requests.post(
        f"{base_url}/triage",
        json={
            "message": message
        },
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def analyse_batch(
    base_url: str,
    uploaded_file,
) -> dict:
    """
    Analyse a CSV batch of incoming messages.
    """

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "text/csv",
        )
    }

    response = requests.post(
        f"{base_url}/analyze",
        files=files,
        timeout=120,
    )

    response.raise_for_status()

    return response.json()


def fetch_cases(
    base_url: str,
    **filters,
) -> list[dict]:
    """
    Retrieve persisted cases from the backend.
    """

    params = {
        key: value
        for key, value in filters.items()
        if value is not None
    }

    response = requests.get(
        f"{base_url}/cases",
        params=params,
        timeout=15,
    )

    response.raise_for_status()

    return response.json()["cases"]


def submit_review(
    base_url: str,
    case_id: int,
    human_urgency: str,
    human_notes: str,
) -> dict:
    """
    Save a human review decision.
    """

    response = requests.post(
        f"{base_url}/cases/{case_id}/review",
        json={
            "human_urgency": human_urgency,
            "human_notes": human_notes,
        },
        timeout=15,
    )

    response.raise_for_status()

    return response.json()


def delete_all_cases(
    base_url: str,
) -> dict:
    """
    Delete all stored test/demo cases.
    """

    response = requests.delete(
        f"{base_url}/cases",
        timeout=15,
    )

    response.raise_for_status()

    return response.json()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _short_time(
    iso_timestamp: str,
) -> str:
    """
    Convert a full ISO timestamp to HH:MM:SS.
    """

    try:
        return datetime.fromisoformat(
            iso_timestamp
        ).strftime(
            "%H:%M:%S"
        )

    except (
        TypeError,
        ValueError,
    ):
        return ""


def confidence_display(
    case: dict,
) -> str:
    """
    Return a suitable confidence display.

    OpenAI structured output does not provide a calibrated probability,
    so displaying 0% or 100% would be misleading.

    Local ML does produce a model probability, so that value can be shown.
    """

    analysis_mode = case.get(
        "analysis_mode",
        "Unknown",
    )

    if analysis_mode == "OpenAI":
        return "N/A"

    confidence = case.get(
        "urgency_confidence"
    )

    if confidence is None:
        return "N/A"

    return f"{confidence:.0%}"


def cases_to_csv(
    cases: list[dict],
) -> str:
    """
    Convert stored cases into downloadable CSV text.
    """

    if not cases:
        return ""

    fieldnames = [
        "id",
        "created_at",
        "message",

        "urgency",
        "urgency_confidence",

        "category",
        "category_confidence",

        "route",
        "escalation",
        "escalate_to_human",

        "analysis_mode",
        "provisional",

        "matched_risk_keywords",
        "attention_language_detected",

        "source",

        "human_reviewed",
        "human_urgency",
        "human_notes",
        "reviewed_at",
    ]

    buffer = io.StringIO()

    writer = csv_module.DictWriter(
        buffer,
        fieldnames=fieldnames,
        extrasaction="ignore",
    )

    writer.writeheader()

    for row in cases:
        export_row = dict(row)

        if isinstance(
            export_row.get(
                "matched_risk_keywords"
            ),
            list,
        ):
            export_row[
                "matched_risk_keywords"
            ] = ", ".join(
                export_row[
                    "matched_risk_keywords"
                ]
            )

        writer.writerow(
            export_row
        )

    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Streamlit state
# ---------------------------------------------------------------------------

def init_state():
    """
    Initialise browser-session UI state.

    Persistent case data remains in SQLite on the backend.
    """

    if (
        "backend_url"
        not in st.session_state
    ):
        st.session_state[
            "backend_url"
        ] = DEFAULT_BACKEND_URL

    if (
        "last_result"
        not in st.session_state
    ):
        st.session_state[
            "last_result"
        ] = None

    if (
        "last_error"
        not in st.session_state
    ):
        st.session_state[
            "last_error"
        ] = None

    if (
        "last_batch_summary"
        not in st.session_state
    ):
        st.session_state[
            "last_batch_summary"
        ] = None

    if (
        "confirm_delete_all"
        not in st.session_state
    ):
        st.session_state[
            "confirm_delete_all"
        ] = False


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HumanFirst AI",
    page_icon="📨",
    layout="centered",
)

init_state()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:

    st.subheader(
        "Backend connection"
    )

    st.session_state[
        "backend_url"
    ] = st.text_input(
        "Backend URL",
        value=st.session_state[
            "backend_url"
        ],
    )

    health = check_backend_health(
        st.session_state[
            "backend_url"
        ]
    )

    if health is None:

        st.error(
            "⚫ Backend unreachable"
        )

    elif health.get(
        "local_model_trained"
    ):

        st.success(
            "🟢 Backend connected — "
            "OpenAI primary + Local ML "
            "fallback ready"
        )

    else:

        st.warning(
            "🟡 Backend connected — "
            "Local ML fallback is not trained"
        )

    st.caption(
        "Past cases are stored on the backend "
        "and persist across browser and "
        "application restarts."
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title(
    "📨 HumanFirst AI"
)

st.caption(
    "AI-assisted government inbox: "
    "urgency detection, routing, "
    "and escalation"
)


tab_analyse, tab_queue, tab_batch, tab_dashboard = st.tabs(
    [
        "Analyse",
        "Priority Queue",
        "Batch Upload",
        "Dashboard",
    ]
)


# ---------------------------------------------------------------------------
# Analyse tab
# ---------------------------------------------------------------------------

with tab_analyse:

    message = st.text_area(
        "Incoming message",
        height=150,
        placeholder=(
            "Paste or type the message "
            "to analyse..."
        ),
        key="message_input",
    )

    analyse_clicked = st.button(
        "Analyse",
        type="primary",
        disabled=not message.strip(),
    )

    st.divider()


    # -----------------------------------------------------------------------
    # Run analysis
    # -----------------------------------------------------------------------

    if analyse_clicked:

        with st.spinner(
            "Analysing..."
        ):

            try:
                result = analyse_message(
                    st.session_state[
                        "backend_url"
                    ],
                    message,
                )

                st.session_state[
                    "last_result"
                ] = result

                st.session_state[
                    "last_error"
                ] = None

            except (
                requests.exceptions.RequestException
            ) as exc:

                st.session_state[
                    "last_result"
                ] = None

                st.session_state[
                    "last_error"
                ] = (
                    "Couldn't analyse the message. "
                    "Make sure the Flask backend is "
                    "running and reachable."
                )


    # -----------------------------------------------------------------------
    # Error display
    # -----------------------------------------------------------------------

    if st.session_state[
        "last_error"
    ]:
        st.error(
            st.session_state[
                "last_error"
            ]
        )


    # -----------------------------------------------------------------------
    # Result display
    # -----------------------------------------------------------------------

    if st.session_state[
        "last_result"
    ]:

        result = st.session_state[
            "last_result"
        ]

        urgency_badge = (
            URGENCY_COLORS.get(
                result[
                    "urgency"
                ],
                "⚪",
            )
        )

        dept_icon = (
            DEPARTMENT_ICONS.get(
                result.get(
                    "category"
                ),
                "📁",
            )
        )

        analysis_mode = (
            result.get(
                "analysis_mode",
                "Unknown",
            )
        )

        provisional = (
            result.get(
                "provisional",
                False,
            )
        )

        st.subheader(
            "AI assessment"
        )

        st.caption(
            f"Case #{result['id']} "
            "— saved to the case store"
        )


        # -------------------------------------------------------------------
        # Main metrics
        # -------------------------------------------------------------------

        col1, col2, col3 = st.columns(
            3
        )

        with col1:
            st.metric(
                "Urgency",
                (
                    f"{urgency_badge} "
                    f"{result['urgency']}"
                ),
            )

        with col2:
            st.metric(
                "Category",
                (
                    f"{dept_icon} "
                    f"{result.get('category', 'Unclassified')}"
                ),
            )

        with col3:
            st.metric(
                "Confidence",
                confidence_display(
                    result
                ),
            )


        # -------------------------------------------------------------------
        # Routing / escalation
        # -------------------------------------------------------------------

        col4, col5, col6 = st.columns(
            3
        )

        with col4:
            st.write(
                "**Route to:** "
                f"{result.get('route', 'Unclassified')}"
            )

        with col5:
            st.write(
                "**Escalation tier:** "
                f"{result.get('escalation', 'No')}"
            )

        with col6:
            st.write(
                "**Flag for human:** "
                + (
                    "Yes"
                    if result.get(
                        "escalate_to_human"
                    )
                    else "No"
                )
            )


        # -------------------------------------------------------------------
        # Analysis mode
        # -------------------------------------------------------------------

        st.write(
            "**Analysis mode:** "
            f"{analysis_mode}"
        )


        if provisional:

            st.warning(
                "⚠️ Provisional result — "
                "the OpenAI primary classifier "
                "was unavailable. The Local ML "
                "fallback was used and this case "
                "requires human review."
            )

        elif (
            analysis_mode
            == "OpenAI"
        ):

            st.caption(
                "OpenAI primary classifier used. "
                "Confidence is shown as N/A because "
                "the structured output does not "
                "provide a calibrated probability."
            )


        # -------------------------------------------------------------------
        # Explanation
        # -------------------------------------------------------------------

        st.write(
            "**Explanation:**"
        )

        st.info(
            result.get(
                "explanation",
                "No explanation available.",
            )
        )


        # -------------------------------------------------------------------
        # Local risk indicators
        # -------------------------------------------------------------------

        if result.get(
            "matched_risk_keywords"
        ):

            st.write(
                "**Risk factors identified:**"
            )

            st.warning(
                ", ".join(
                    result[
                        "matched_risk_keywords"
                    ]
                )
            )


        if result.get(
            "attention_language_detected"
        ):

            st.caption(
                "⚠️ Urgent-sounding language "
                "was detected. This alone does "
                "not determine the urgency level."
            )


        # -------------------------------------------------------------------
        # Human review
        # -------------------------------------------------------------------

        st.divider()

        st.subheader(
            "Human review"
        )


        if result.get(
            "human_reviewed"
        ):

            reviewed_message = (
                "Already reviewed: "
                "final urgency "
                f"**{result['human_urgency']}**"
            )

            if result.get(
                "human_notes"
            ):
                reviewed_message += (
                    " — "
                    f"\"{result['human_notes']}\""
                )

            st.success(
                reviewed_message
            )

        else:

            st.caption(
                "Confirm or override the AI's "
                "recommendation. The human decision "
                "is saved to the case permanently."
            )

            urgency_options = [
                "Critical",
                "High",
                "Normal",
            ]

            selected_index = (
                urgency_options.index(
                    result["urgency"]
                )
                if result[
                    "urgency"
                ] in urgency_options
                else 2
            )

            human_urgency = (
                st.selectbox(
                    "Final urgency "
                    "(defaults to the AI's call)",
                    options=urgency_options,
                    index=selected_index,
                    key=(
                        f"human_urgency_"
                        f"{result['id']}"
                    ),
                )
            )

            human_notes = (
                st.text_input(
                    "Reviewer notes (optional)",
                    key=(
                        f"human_notes_"
                        f"{result['id']}"
                    ),
                )
            )


            # ---------------------------------------------------------------
            # Draft acknowledgement
            # ---------------------------------------------------------------

            draft = ACK_TEMPLATES.get(
                human_urgency,
                ACK_TEMPLATES[
                    "Normal"
                ],
            )

            with st.expander(
                "Draft acknowledgement "
                "(review before sending)"
            ):

                st.text_area(
                    "Draft",
                    value=draft,
                    height=100,
                    key=(
                        f"draft_ack_"
                        f"{result['id']}"
                    ),
                )

                st.caption(
                    "This is only a draft. "
                    "Human staff decide whether "
                    "and how it should be used."
                )


            # ---------------------------------------------------------------
            # Save review
            # ---------------------------------------------------------------

            if st.button(
                "Confirm & save this decision",
                type="primary",
                key=(
                    f"save_review_"
                    f"{result['id']}"
                ),
            ):

                try:
                    updated = submit_review(
                        st.session_state[
                            "backend_url"
                        ],
                        result["id"],
                        human_urgency,
                        human_notes,
                    )

                    st.session_state[
                        "last_result"
                    ] = updated

                    st.success(
                        f"Human decision saved "
                        f"to case #{updated['id']}."
                    )

                    st.rerun()

                except (
                    requests.exceptions.RequestException
                ) as exc:

                    st.error(
                        "Couldn't save the human "
                        f"review: {exc}"
                    )


    elif not st.session_state[
        "last_error"
    ]:

        st.write(
            "Enter a message above and click "
            "**Analyse** to see the result."
        )


# ---------------------------------------------------------------------------
# Priority Queue tab
# ---------------------------------------------------------------------------

with tab_queue:

    st.caption(
        "Stored cases are shown most urgent first. "
        "OpenAI is the primary classifier; Local ML "
        "results are clearly marked as provisional."
    )

    try:
        cases = fetch_cases(
            st.session_state[
                "backend_url"
            ]
        )

    except (
        requests.exceptions.RequestException
    ):
        cases = None

        st.error(
            "Couldn't retrieve cases "
            "from the backend."
        )


    if cases is not None:

        if not cases:

            st.write(
                "No cases stored yet. "
                "Analyse a message or upload "
                "a batch CSV."
            )

        else:

            queue = sorted(
                cases,
                key=lambda case: (
                    URGENCY_RANK.get(
                        case.get(
                            "human_urgency"
                        )
                        if case.get(
                            "human_reviewed"
                        )
                        else case.get(
                            "urgency"
                        ),
                        99,
                    ),

                    0
                    if case.get(
                        "escalate_to_human"
                    )
                    else 1,

                    -(
                        case.get(
                            "urgency_confidence"
                        )
                        or 0
                    ),
                ),
            )


            rows = []

            for item in queue:

                final_urgency = (
                    item.get(
                        "human_urgency"
                    )
                    if item.get(
                        "human_reviewed"
                    )
                    else item.get(
                        "urgency"
                    )
                )

                urgency_badge = (
                    URGENCY_COLORS.get(
                        final_urgency,
                        "⚪",
                    )
                )

                rows.append(
                    {
                        "":
                            urgency_badge,

                        "ID":
                            item["id"],

                        "Urgency":
                            final_urgency,

                        "Category":
                            item.get(
                                "category"
                            ),

                        "Route":
                            item.get(
                                "route"
                            ),

                        "Analysis mode":
                            item.get(
                                "analysis_mode",
                                "Unknown",
                            ),

                        "Provisional":
                            (
                                "Yes"
                                if item.get(
                                    "provisional"
                                )
                                else "No"
                            ),

                        "Escalate?":
                            (
                                "Yes"
                                if item.get(
                                    "escalate_to_human"
                                )
                                else "No"
                            ),

                        "Confidence":
                            confidence_display(
                                item
                            ),

                        "Reviewed?":
                            (
                                "Yes"
                                if item.get(
                                    "human_reviewed"
                                )
                                else "No"
                            ),

                        "Message":
                            (
                                item[
                                    "message"
                                ][:80]
                                + (
                                    "..."
                                    if len(
                                        item[
                                            "message"
                                        ]
                                    ) > 80
                                    else ""
                                )
                            ),

                        "Time":
                            _short_time(
                                item.get(
                                    "created_at"
                                )
                            ),
                    }
                )


            st.dataframe(
                rows,
                use_container_width=True,
                hide_index=True,
            )


# ---------------------------------------------------------------------------
# Batch Upload tab
# ---------------------------------------------------------------------------

with tab_batch:

    st.write(
        "Upload a CSV containing a `message` column. "
        "Each message is analysed through the same "
        "OpenAI-primary / Local-ML-fallback pipeline "
        "and saved to the case store."
    )

    uploaded_file = st.file_uploader(
        "CSV file",
        type=[
            "csv"
        ],
    )


    if (
        uploaded_file
        and st.button(
            "Analyse batch",
            type="primary",
        )
    ):

        with st.spinner(
            f"Analysing "
            f"{uploaded_file.name}..."
        ):

            try:
                batch_result = (
                    analyse_batch(
                        st.session_state[
                            "backend_url"
                        ],
                        uploaded_file,
                    )
                )

                st.session_state[
                    "last_batch_summary"
                ] = batch_result[
                    "summary"
                ]

                st.success(
                    "Analysed and saved "
                    f"{batch_result['summary']['total_messages']} "
                    "messages."
                )

            except (
                requests.exceptions.RequestException
            ) as exc:

                st.error(
                    "Batch analysis failed: "
                    f"{exc}"
                )


    if st.session_state[
        "last_batch_summary"
    ]:

        st.subheader(
            "Last batch summary"
        )

        summary = st.session_state[
            "last_batch_summary"
        ]

        col1, col2 = st.columns(
            2
        )

        col1.metric(
            "Total messages",
            summary[
                "total_messages"
            ],
        )

        col2.metric(
            "Escalated",
            (
                f"{summary['escalated_count']} "
                f"({summary['escalation_rate']:.0%})"
            ),
        )

        st.write(
            "**By urgency:**",
            summary[
                "by_urgency"
            ],
        )

        st.write(
            "**By category:**",
            summary[
                "by_category"
            ],
        )


# ---------------------------------------------------------------------------
# Dashboard tab
# ---------------------------------------------------------------------------

with tab_dashboard:

    try:
        cases = fetch_cases(
            st.session_state[
                "backend_url"
            ]
        )

    except (
        requests.exceptions.RequestException
    ):
        cases = None

        st.error(
            "Couldn't retrieve dashboard "
            "data from the backend."
        )


    if cases is not None:

        if not cases:

            st.write(
                "No cases stored yet. "
                "Analyse a message or upload "
                "a batch to populate the dashboard."
            )

        else:

            total = len(
                cases
            )

            critical = sum(
                1
                for case in cases
                if (
                    case.get(
                        "human_urgency"
                    )
                    if case.get(
                        "human_reviewed"
                    )
                    else case.get(
                        "urgency"
                    )
                )
                == "Critical"
            )

            high = sum(
                1
                for case in cases
                if (
                    case.get(
                        "human_urgency"
                    )
                    if case.get(
                        "human_reviewed"
                    )
                    else case.get(
                        "urgency"
                    )
                )
                == "High"
            )

            normal = sum(
                1
                for case in cases
                if (
                    case.get(
                        "human_urgency"
                    )
                    if case.get(
                        "human_reviewed"
                    )
                    else case.get(
                        "urgency"
                    )
                )
                == "Normal"
            )

            escalated = sum(
                1
                for case in cases
                if case.get(
                    "escalate_to_human"
                )
            )

            escalation_rate = (
                (
                    escalated
                    / total
                )
                * 100
                if total
                else 0
            )

            reviewed = sum(
                1
                for case in cases
                if case.get(
                    "human_reviewed"
                )
            )

            overridden = sum(
                1
                for case in cases
                if (
                    case.get(
                        "human_reviewed"
                    )
                    and
                    case.get(
                        "human_urgency"
                    )
                    != case.get(
                        "urgency"
                    )
                )
            )

            openai_count = sum(
                1
                for case in cases
                if case.get(
                    "analysis_mode"
                )
                == "OpenAI"
            )

            fallback_count = sum(
                1
                for case in cases
                if case.get(
                    "analysis_mode"
                )
                == "Local ML"
            )

            provisional_count = sum(
                1
                for case in cases
                if case.get(
                    "provisional"
                )
            )


            # ---------------------------------------------------------------
            # Local ML confidence only
            # ---------------------------------------------------------------

            local_confidences = [
                case[
                    "urgency_confidence"
                ]
                for case in cases
                if (
                    case.get(
                        "analysis_mode"
                    )
                    == "Local ML"
                    and
                    case.get(
                        "urgency_confidence"
                    )
                    is not None
                )
            ]

            avg_local_confidence = (
                sum(
                    local_confidences
                )
                / len(
                    local_confidences
                )
                if local_confidences
                else None
            )


            # ---------------------------------------------------------------
            # Main summary
            # ---------------------------------------------------------------

            st.subheader(
                "Summary"
            )

            col1, col2, col3 = (
                st.columns(
                    3
                )
            )

            col1.metric(
                "Total analysed",
                total,
            )

            col2.metric(
                "Escalated to human",
                (
                    f"{escalated} "
                    f"({escalation_rate:.0f}%)"
                ),
            )

            col3.metric(
                "Human reviewed",
                reviewed,
            )


            col4, col5, col6 = (
                st.columns(
                    3
                )
            )

            col4.metric(
                "OpenAI primary",
                openai_count,
            )

            col5.metric(
                "Local ML fallback",
                fallback_count,
            )

            col6.metric(
                "Provisional cases",
                provisional_count,
            )


            col7, col8 = (
                st.columns(
                    2
                )
            )

            col7.metric(
                "Human overrode AI",
                overridden,
            )

            col8.metric(
                "Avg. Local ML confidence",
                (
                    f"{avg_local_confidence:.0%}"
                    if avg_local_confidence
                    is not None
                    else "N/A"
                ),
            )


            # ---------------------------------------------------------------
            # Urgency breakdown
            # ---------------------------------------------------------------

            st.subheader(
                "Urgency breakdown"
            )

            col_a, col_b, col_c = (
                st.columns(
                    3
                )
            )

            col_a.metric(
                "🔴 Critical",
                critical,
            )

            col_b.metric(
                "🟠 High",
                high,
            )

            col_c.metric(
                "🟢 Normal",
                normal,
            )

            st.bar_chart(
                {
                    "Critical":
                        critical,

                    "High":
                        high,

                    "Normal":
                        normal,
                }
            )


            # ---------------------------------------------------------------
            # Category breakdown
            # ---------------------------------------------------------------

            st.subheader(
                "Category breakdown"
            )

            category_counts = {
                department: 0
                for department
                in DEPARTMENT_ICONS
            }

            for case in cases:

                category = case.get(
                    "category"
                )

                if (
                    category
                    in category_counts
                ):
                    category_counts[
                        category
                    ] += 1

            st.bar_chart(
                category_counts
            )


            # ---------------------------------------------------------------
            # Recent cases
            # ---------------------------------------------------------------

            st.subheader(
                "Recent cases"
            )

            for item in cases[:10]:

                final_urgency = (
                    item.get(
                        "human_urgency"
                    )
                    if item.get(
                        "human_reviewed"
                    )
                    else item.get(
                        "urgency"
                    )
                )

                urgency_badge = (
                    URGENCY_COLORS.get(
                        final_urgency,
                        "⚪",
                    )
                )

                dept_icon = (
                    DEPARTMENT_ICONS.get(
                        item.get(
                            "category"
                        ),
                        "📁",
                    )
                )

                mode = item.get(
                    "analysis_mode",
                    "Unknown",
                )

                provisional_flag = (
                    " · provisional"
                    if item.get(
                        "provisional"
                    )
                    else ""
                )

                reviewed_flag = (
                    " · human reviewed"
                    if item.get(
                        "human_reviewed"
                    )
                    else ""
                )

                override_flag = (
                    " · overridden"
                    if (
                        item.get(
                            "human_reviewed"
                        )
                        and
                        item.get(
                            "human_urgency"
                        )
                        != item.get(
                            "urgency"
                        )
                    )
                    else ""
                )

                confidence_text = (
                    confidence_display(
                        item
                    )
                )

                st.write(
                    f"#{item['id']} "
                    f"{urgency_badge} "
                    f"**{final_urgency}** "
                    f"· {dept_icon} "
                    f"{item.get('category', 'Unclassified')} "
                    f"· {mode}"
                    f"{provisional_flag}"
                    f"{reviewed_flag}"
                    f"{override_flag} "
                    f"· confidence {confidence_text} "
                    f"· {_short_time(item.get('created_at'))}"
                )


            # ---------------------------------------------------------------
            # Export / delete
            # ---------------------------------------------------------------

            st.divider()

            col_export, col_clear = (
                st.columns(
                    2
                )
            )


            with col_export:

                csv_data = (
                    cases_to_csv(
                        cases
                    )
                )

                st.download_button(
                    "Export all cases as CSV",
                    data=csv_data,
                    file_name=(
                        "humanfirst_cases_"
                        + datetime.now().strftime(
                            "%Y%m%d_%H%M%S"
                        )
                        + ".csv"
                    ),
                    mime="text/csv",
                )


            with col_clear:

                st.session_state[
                    "confirm_delete_all"
                ] = st.checkbox(
                    "I understand this permanently "
                    "deletes ALL stored cases",
                    value=st.session_state[
                        "confirm_delete_all"
                    ],
                )

                if st.button(
                    "Delete all stored cases",
                    disabled=not (
                        st.session_state[
                            "confirm_delete_all"
                        ]
                    ),
                ):

                    try:
                        result = (
                            delete_all_cases(
                                st.session_state[
                                    "backend_url"
                                ]
                            )
                        )

                        st.session_state[
                            "confirm_delete_all"
                        ] = False

                        st.session_state[
                            "last_batch_summary"
                        ] = None

                        st.session_state[
                            "last_result"
                        ] = None

                        st.success(
                            "Deleted "
                            f"{result['deleted']} "
                            "stored cases."
                        )

                        st.rerun()

                    except (
                        requests.exceptions.RequestException
                    ) as exc:

                        st.error(
                            "Couldn't delete "
                            f"cases: {exc}"
                        )