import os
from datetime import datetime, timezone

from pathlib import Path

import pandas as pd
import requests
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="IBVAP | Intelligent Border Video Analytics Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_API_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000"
)

API_BASE_URL = DEFAULT_API_URL.rstrip("/")


# ============================================================
# VIDEO PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

VIDEO_DIR = BASE_DIR / "assets" / "videos"

# Original camera videos
CAMERA1_VIDEO = VIDEO_DIR / "camera1.mp4"
CAMERA2_VIDEO = VIDEO_DIR / "camera2.mp4"

# YOLO detected videos
CAMERA1_DETECTED_VIDEO = VIDEO_DIR / "camera1_detected.mp4"
CAMERA2_DETECTED_VIDEO = VIDEO_DIR / "camera2_detected.mp4"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0b1220;
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    .main-header {
        background: linear-gradient(135deg, #172033, #202d42);
        padding: 2.2rem;
        border-radius: 18px;
        border: 1px solid #2d3a50;
        margin-bottom: 1.5rem;
    }

    .platform-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #5bb8f0;
        margin-bottom: 0.3rem;
    }

    .platform-subtitle {
        font-size: 1rem;
        color: #aab7c8;
    }

    .metric-card {
        background: #111827;
        border: 1px solid #2a3a52;
        border-radius: 14px;
        padding: 1rem;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.7rem;
        color: #f1f5f9;
    }

    .small-label {
        color: #94a3b8;
        font-size: 0.85rem;
    }

    .status-online {
        color: #22c55e;
        font-weight: 700;
    }

    .status-offline {
        color: #ef4444;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API HELPERS
# ============================================================

def api_get(endpoint, timeout=10):
    try:
        response = requests.get(
            f"{API_BASE_URL}{endpoint}",
            timeout=timeout,
        )
        response.raise_for_status()

        return response.json(), None

    except requests.RequestException as error:
        return None, str(error)


def api_put(endpoint, timeout=10):
    try:
        response = requests.put(
            f"{API_BASE_URL}{endpoint}",
            timeout=timeout,
        )

        response.raise_for_status()

        return response.json(), None

    except requests.RequestException as error:
        return None, str(error)


def check_backend():

    try:

        response = requests.get(
            f"{API_BASE_URL}/api/v1/health",
            timeout=5,
        )

        response.raise_for_status()

        return True

    except requests.RequestException:

        return False


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        # 🛰️ IBVAP Link

        **Intelligent Border Video Analytics Platform**
        """
    )

    st.divider()

    api_url_input = st.text_input(
        "Central API URL",
        value=API_BASE_URL,
    )

    API_BASE_URL = api_url_input.rstrip("/")

    st.divider()

    auto_refresh = st.checkbox(
        "Enable Live Auto-Refresh",
        value=False,
    )

    refresh_interval = st.slider(
        "Auto-Refresh Interval (seconds)",
        min_value=5,
        max_value=60,
        value=10,
    )

    st.divider()

    st.subheader("🎯 Tactical Filters")

    priority_filter = st.selectbox(
        "Priority Level",
        [
            "ALL",
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL",
        ],
    )

    review_filter = st.selectbox(
        "Operator Review",
        [
            "ALL",
            "UNRESOLVED",
            "RESOLVED",
        ],
    )

    camera_designation = st.text_input(
        "Camera Designation",
        placeholder="e.g. CAM-01",
    )

    st.divider()

    st.caption(
        "IBVAP • Tactical Threat Command Center"
    )


# ============================================================
# BACKEND STATUS
# ============================================================

backend_online = check_backend()


# ============================================================
# HEADER
# ============================================================

current_time = datetime.now(
    timezone.utc
).strftime("%Y-%m-%d %H:%M:%S UTC")


left_header, right_header = st.columns(
    [4, 1]
)


with left_header:

    st.markdown(
        """
        <div class="main-header">
            <div class="platform-title">
                🛰️ Intelligent Border Video Analytics Platform
            </div>

        </div>
        """,
    unsafe_allow_html = True,
    )


with right_header:

    st.markdown(
        "### SYSTEM TIME"
    )

    st.code(
        current_time
    )


# ============================================================
# LOAD DASHBOARD DATA
# ============================================================

summary = {}

if backend_online:

    summary_data, summary_error = api_get(
        "/api/v1/dashboard/summary"
    )

    if not summary_error and summary_data:

        summary = summary_data


alerts_data = {}

if backend_online:

    alerts_response, alerts_error = api_get(
        "/api/v1/alerts/?unresolved_only=true"
    )

    if not alerts_error and alerts_response:

        alerts_data = alerts_response


active_alerts = alerts_data.get(
    "total",
    0,
)


# ============================================================
# TOP METRICS
# ============================================================

metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(
    5
)


with metric_1:

    st.metric(
        "🔗 API LINK STATUS",
        "ONLINE" if backend_online else "OFFLINE",
    )


with metric_2:

    st.metric(
        "👁 TOTAL DETECTIONS",
        summary.get(
            "total_detection_events",
            0,
        ),
    )


with metric_3:

    st.metric(
        "🚨 ACTIVE ALERTS",
        active_alerts,
    )


with metric_4:

    st.metric(
        "📹 ACTIVE CAMERAS",
        summary.get(
            "active_cameras",
            0,
        ),
    )


with metric_5:

    st.metric(
        "🕒 LAST 24 HOURS",
        summary.get(
            "detections_last_24_hours",
            0,
        ),
    )


st.divider()


# ============================================================
# MAIN TABS
# ============================================================

tab_live, tab_detection, tab_alerts, tab_analytics, tab_cameras = st.tabs(
    [
        "📹 Live Feeds",
        "🤖 AI Detection",
        "🚨 Alert Queue",
        "📊 Analytics",
        "⚙️ Camera Config",
    ]
)


# ============================================================
# TAB 1 — LIVE FEEDS
# ============================================================

with tab_live:

    # ============================================================
    # LIVE SURVEILLANCE WALL
    # ============================================================

    st.markdown("## 📹 Live Surveillance Wall")

    st.caption(
        "Operational surveillance feeds • YOLO AI Detection • Tactical monitoring"
    )

    cam1_col, cam2_col = st.columns(2)


    # ============================================================
    # CAMERA 1
    # ============================================================

    with cam1_col:

        st.markdown("## CAM-01")

        # Use YOLO detected video if available
        if CAMERA1_DETECTED_VIDEO.exists():

            st.video(
                str(CAMERA1_DETECTED_VIDEO)
            )

            st.success(
                "🟢 CAM-01 • YOLO AI Detection Active"
            )

        # Otherwise show original video
        elif CAMERA1_VIDEO.exists():

            st.video(
                str(CAMERA1_VIDEO)
            )

            st.info(
                "🔵 CAM-01 • Original surveillance feed"
            )

        else:

            st.error(
                "CAM-01 video not found."
            )


    # ============================================================
    # CAMERA 2
    # ============================================================

    with cam2_col:

        st.markdown("## CAM-02")

        # Use YOLO detected video if available
        if CAMERA2_DETECTED_VIDEO.exists():

            st.video(
                str(CAMERA2_DETECTED_VIDEO)
            )

            st.success(
                "🟢 CAM-02 • YOLO AI Detection Active"
            )

        # Otherwise show original video
        elif CAMERA2_VIDEO.exists():

            st.video(
                str(CAMERA2_VIDEO)
            )

            st.info(
                "🔵 CAM-02 • Original surveillance feed"
            )

        else:

            st.error(
                "CAM-02 video not found."
            )


    st.divider()

    if backend_online:

        events, events_error = api_get(
            "/api/v1/detection-events/"
        )

        if not events_error and events:

            st.subheader(
                "🛰️ Recent AI Detection Activity"
            )

            event_df = pd.DataFrame(
                events
            )

            display_columns = [
                column
                for column in [
                    "id",
                    "camera_id",
                    "object_type",
                    "confidence",
                    "detected_at",
                ]
                if column in event_df.columns
            ]

            if "confidence" in event_df.columns:

                event_df["confidence"] = (
                    event_df["confidence"] * 100
                ).round(2)

            st.dataframe(
                event_df[
                    display_columns
                ],
                width="stretch",
                hide_index=True,
            )


# ============================================================
# TAB 2 — AI DETECTION
# ============================================================

with tab_detection:

    st.markdown(
        '<div class="section-title">🤖 AI Image Intelligence</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Upload surveillance imagery and run YOLO-powered object detection"
    )


    uploaded_file = st.file_uploader(
        "Upload Surveillance Image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
    )


    confidence = st.slider(
        "Detection Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )


    if uploaded_file:

        image_col, result_col = st.columns(
            2
        )


        with image_col:

            st.subheader(
                "📷 Evidence Input"
            )

            st.image(
                uploaded_file,
                caption=uploaded_file.name,
                width="stretch",
            )


        with result_col:

            st.subheader(
                "🧠 AI Detection Engine"
            )

            if st.button(
                "🚀 RUN AI DETECTION",
                width="stretch",
            ):

                try:

                    with st.spinner(
                        "YOLO AI is analysing the surveillance image..."
                    ):

                        files = {
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type,
                            )
                        }


                        response = requests.post(
                            f"{API_BASE_URL}/api/v1/ai/detect/image",
                            files=files,
                            params={
                                "confidence_threshold": confidence
                            },
                            timeout=90,
                        )


                        response.raise_for_status()


                        result = response.json()


                    total_detections = result.get(
                        "total_detections",
                        0,
                    )


                    st.success(
                        "AI detection completed successfully."
                    )


                    st.metric(
                        "Objects Detected",
                        total_detections,
                    )


                    detections = result.get(
                        "detections",
                        [],
                    )


                    if detections:

                        detection_rows = []


                        for detection in detections:

                            detection_rows.append(
                                {
                                    "Object": detection.get(
                                        "object_type",
                                        "Unknown",
                                    ),
                                    "Confidence (%)": round(
                                        detection.get(
                                            "confidence",
                                            0,
                                        ) * 100,
                                        2,
                                    ),
                                    "Detected At": detection.get(
                                        "detected_at"
                                    ),
                                }
                            )


                        detection_df = pd.DataFrame(
                            detection_rows
                        )


                        st.dataframe(
                            detection_df,
                            width="stretch",
                            hide_index=True,
                        )


                    else:

                        st.info(
                            "No objects were detected."
                        )


                except requests.RequestException as error:

                    st.error(
                        f"Detection request failed: {error}"
                    )


# ============================================================
# TAB 3 — ALERT QUEUE
# ============================================================

with tab_alerts:

    st.markdown(
        '<div class="section-title">🚨 Tactical Alert Queue</div>',
        unsafe_allow_html=True,
    )

    alerts_data, alerts_error = api_get(
        "/api/v1/alerts/?unresolved_only=true"
    )


    if alerts_error:

        st.warning(
            f"Unable to load alerts: {alerts_error}"
        )


    else:

        alerts = alerts_data.get(
            "alerts",
            [],
        )


        if priority_filter != "ALL":

            alerts = [
                alert
                for alert in alerts
                if alert.get(
                    "alert_level"
                ) == priority_filter
            ]


        if alerts:

            for alert in alerts:

                alert_id = alert.get(
                    "id"
                )


                with st.container(
                    border=True
                ):

                    left, right = st.columns(
                        [5, 1]
                    )


                    with left:

                        st.subheader(
                            f"🚨 {alert.get('alert_level', 'UNKNOWN')} ALERT"
                        )

                        st.write(
                            alert.get(
                                "message",
                                "No message available.",
                            )
                        )

                        st.caption(
                            f"Camera ID: {alert.get('camera_id')}"
                        )

                        st.caption(
                            f"Detection Event: "
                            f"{alert.get('detection_event_id')}"
                        )

                        st.caption(
                            f"Created: "
                            f"{alert.get('created_at')}"
                        )


                    with right:

                        if st.button(
                            "✓ RESOLVE",
                            key=f"resolve_{alert_id}",
                            width="stretch",
                        ):

                            result, error = api_put(
                                f"/api/v1/alerts/"
                                f"{alert_id}/resolve"
                            )


                            if error:

                                st.error(
                                    f"Resolve failed: {error}"
                                )


                            else:

                                st.success(
                                    "Alert resolved."
                                )

                                st.rerun()


        else:

            st.success(
                "🟢 No unresolved security alerts."
            )


# ============================================================
# TAB 4 — ANALYTICS
# ============================================================

with tab_analytics:

    st.markdown(
        '<div class="section-title">📊 Detection Intelligence Analytics</div>',
        unsafe_allow_html=True,
    )


    analytics, analytics_error = api_get(
        "/api/v1/dashboard/detections-by-type"
    )


    if analytics_error:

        st.warning(
            "Analytics could not be loaded."
        )


    else:

        detection_types = analytics.get(
            "detections",
            [],
        )


        if detection_types:

            analytics_df = pd.DataFrame(
                detection_types
            )


            chart_1, chart_2 = st.columns(
                2
            )


            with chart_1:

                st.subheader(
                    "Object Distribution"
                )

                st.bar_chart(
                    analytics_df.set_index(
                        "object_type"
                    )["count"]
                )


            with chart_2:

                st.subheader(
                    "Average Confidence"
                )

                confidence_df = (
                    analytics_df.copy()
                )

                confidence_df[
                    "average_confidence"
                ] = (
                    confidence_df[
                        "average_confidence"
                    ] * 100
                )


                st.bar_chart(
                    confidence_df.set_index(
                        "object_type"
                    )[
                        "average_confidence"
                    ]
                )


        else:

            st.info(
                "No analytics data available yet."
            )


    st.divider()


    st.subheader(
        "🕒 Detection Timeline"
    )


    events, events_error = api_get(
        "/api/v1/detection-events/"
    )


    if not events_error and events:

        timeline_df = pd.DataFrame(
            events
        )


        if (
            "detected_at" in timeline_df.columns
            and "confidence" in timeline_df.columns
        ):

            timeline_df[
                "detected_at"
            ] = pd.to_datetime(
                timeline_df[
                    "detected_at"
                ]
            )


            timeline_df = timeline_df.sort_values(
                "detected_at"
            )


            st.line_chart(
                timeline_df.set_index(
                    "detected_at"
                )[
                    "confidence"
                ]
            )


# ============================================================
# TAB 5 — CAMERA CONFIG
# ============================================================

with tab_cameras:

    st.markdown(
        '<div class="section-title">⚙️ Camera Configuration</div>',
        unsafe_allow_html=True,
    )


    cameras, cameras_error = api_get(
        "/api/v1/cameras/"
    )


    if cameras_error:

        st.warning(
            "Camera data could not be loaded."
        )


    else:

        if cameras:

            camera_df = pd.DataFrame(
                cameras
            )


            st.dataframe(
                camera_df,
                width="stretch",
                hide_index=True,
            )


        else:

            st.info(
                "No cameras configured."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "IBVAP • Intelligent Border Video Analytics Platform • "
    "AI-Powered Tactical Surveillance System"
)


# ============================================================
# OPTIONAL AUTO REFRESH
# ============================================================

if auto_refresh:

    st.caption(
        f"Live refresh enabled every {refresh_interval} seconds."
    )