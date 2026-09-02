import streamlit as st
import requests
import json
import time
import base64
from datetime import datetime
from typing import Optional, Dict, List, Any
import folium
from streamlit_folium import st_folium

# Configure Streamlit page
st.set_page_config(
    page_title="Intelligent Border Video Analytics Platform (IBVAP)",
    page_icon="https://cdn-icons-png.flaticon.com/128/2092/2092661.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 404 Page Logic (Streamlit SPA routing simulation)
try:
    params = st.query_params
    page = params.get("page", "")
except AttributeError:
    params = st.experimental_get_query_params()
    page = params.get("page", [""])[0]

if page not in ["", "dashboard"]:
    st.markdown("<h1 style='text-align: center; margin-top: 20vh; font-size: 72px; color: #ef4444;'><i class='fa-solid fa-triangle-exclamation'></i> 404</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #94a3b8;'>Tactical Sector Not Found</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b;'>The requested dashboard route does not exist.</p>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; margin-top: 20px;'><a href='/?page=dashboard' style='color: #38bdf8; text-decoration: none; border: 1px solid #38bdf8; padding: 10px 20px; border-radius: 8px;'>Return to Command Center</a></div>", unsafe_allow_html=True)
    st.stop()

# Premium CSS Overhaul with Glassmorphism, Micro-animations, and Modern Typography
st.markdown(
    """
    <meta name="description" content="Intelligent Border Video Analytics Platform (IBVAP) - AI-powered tactical command dashboard for border surveillance.">
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    body, .stApp { background: radial-gradient(circle at top, #1e293b 0%, #020617 100%); background-attachment: fixed; color: #f8fafc; font-family: 'Inter', sans-serif; overflow-x: hidden !important; }
    header[data-testid="stHeader"] { background-color: transparent !important; }
    footer { display: none; }
    .stDeployButton { display: none; }
    .block-container { padding-bottom: 0rem !important; }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }
    .header-container { background: rgba(30, 41, 59, 0.4); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 24px 32px; margin-bottom: 28px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5); display: flex; justify-content: space-between; align-items: center; }
    .header-title { font-size: 28px; font-weight: 800; background: linear-gradient(to right, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.5px; margin: 0; }
    .header-subtitle { font-size: 14px; color: #94a3b8; margin-top: 6px; font-weight: 500; letter-spacing: 0.5px; }
    .kpi-card { background: rgba(15, 23, 42, 0.6); backdrop-filter: blur(8px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 20px; text-align: center; transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2); }
    .kpi-card:hover { transform: translateY(-4px); box-shadow: 0 12px 20px -8px rgba(0, 0, 0, 0.4); border-color: rgba(255, 255, 255, 0.2); }
    .kpi-val { font-size: 36px; font-weight: 800; margin: 4px 0; line-height: 1; letter-spacing: -1px; }
    .kpi-lbl { font-size: 12px; text-transform: uppercase; color: #94a3b8; font-weight: 600; letter-spacing: 1px; }
    .alert-card { background: linear-gradient(145deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.98)); backdrop-filter: blur(12px); border: 1px solid rgba(148, 163, 184, 0.3); border-left: 6px solid #475569; border-radius: 12px; padding: 18px; margin-bottom: 24px; transition: all 0.3s ease; box-shadow: 0 8px 16px -4px rgba(0, 0, 0, 0.4); }
    .alert-card:hover { transform: translateX(6px); border-color: rgba(255, 255, 255, 0.4); box-shadow: 0 15px 30px -5px rgba(0, 0, 0, 0.6); }
    .card-critical { border-left-color: #ef4444; }
    .card-high { border-left-color: #f97316; }
    .card-medium { border-left-color: #eab308; }
    .card-low { border-left-color: #22c55e; }
    .compressed-img { object-fit: cover; width: 100%; max-height: 160px; border-radius: 8px; } /* width controlled inline */
    .badge { padding: 6px 12px; border-radius: 6px; font-weight: 700; font-size: 10px; letter-spacing: 1px; text-transform: uppercase; }
    .badge-critical { background: #7f1d1d; color: #fecaca; border: 1px solid #ef4444; box-shadow: 0 0 12px rgba(239, 68, 68, 0.4); }
    .badge-high { background: #7c2d12; color: #fed7aa; border: 1px solid #f97316; box-shadow: 0 0 12px rgba(249, 115, 22, 0.4); }
    .badge-medium { background: #713f12; color: #fef08a; border: 1px solid #eab308; box-shadow: 0 0 12px rgba(234, 179, 8, 0.4); }
    .badge-low { background: #14532d; color: #bbf7d0; border: 1px solid #22c55e; box-shadow: 0 0 12px rgba(34, 197, 94, 0.4); }
    .telemetry-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin-top: 12px; }
    .telemetry-item { background: rgba(0, 0, 0, 0.3); border: 1px solid rgba(255, 255, 255, 0.05); padding: 8px 10px; border-radius: 6px; display: flex; flex-direction: column; }
    .telemetry-lbl { font-size: 9px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px; }
    .telemetry-val { font-size: 12px; font-weight: 600; color: #f1f5f9; font-family: 'Inter', monospace; }
    div[data-testid="stButton"] > button { width: 100%; border-radius: 6px; font-weight: 600; height: 36px; font-size: 12px; border: none; transition: all 0.2s ease; }
    div[data-testid="column"]:nth-child(1) div[data-testid="stButton"] > button { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: white; box-shadow: 0 4px 10px rgba(239, 68, 68, 0.3); }
    div[data-testid="column"]:nth-child(1) div[data-testid="stButton"] > button:hover { background: linear-gradient(135deg, #f87171 0%, #ef4444 100%); transform: translateY(-2px); }
    div[data-testid="column"]:nth-child(2) div[data-testid="stButton"] > button { background: rgba(51, 65, 85, 0.8); color: #e2e8f0; border: 1px solid #475569; }
    div[data-testid="column"]:nth-child(2) div[data-testid="stButton"] > button:hover { background: rgba(71, 85, 105, 1); border-color: #64748b; transform: translateY(-2px); }
    section[data-testid="stSidebar"] { background: rgba(15, 23, 42, 0.8) !important; backdrop-filter: blur(16px); border-right: 1px solid rgba(255, 255, 255, 0.05); }
    iframe { border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5); }
    /* Phase 5: CCTV Wall */
    .cctv-tile-wrapper { position: relative; background: #000; border-radius: 10px; overflow: hidden; border: 1px solid rgba(56, 189, 248, 0.25); box-shadow: 0 0 18px rgba(0,0,0,0.6); transition: border-color 0.3s ease, box-shadow 0.3s ease; }
    .cctv-tile-wrapper:hover { border-color: rgba(56, 189, 248, 0.6); box-shadow: 0 0 24px rgba(56, 189, 248, 0.15); }
    .cctv-tile-wrapper img { width: 100%; aspect-ratio: 16/9; object-fit: cover; display: block; pointer-events: none; user-select: none; -webkit-user-drag: none; }
    .cctv-label { position: absolute; top: 10px; left: 10px; background: rgba(2, 6, 23, 0.75); backdrop-filter: blur(6px); border: 1px solid rgba(56, 189, 248, 0.3); color: #38bdf8; font-size: 10px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; padding: 4px 10px; border-radius: 4px; }
    .cctv-live-dot { position: absolute; top: 12px; right: 12px; width: 8px; height: 8px; background: #22c55e; border-radius: 50%; animation: pulse-live 1.8s infinite; }
    @keyframes pulse-live { 0% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7); } 70% { box-shadow: 0 0 0 6px rgba(34, 197, 94, 0); } 100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); } }
    .zone-preview-box { background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 10px; padding: 16px; margin-top: 12px; }
    div[data-testid="stSegmentedControl"] { background: rgba(15, 23, 42, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px; padding: 4px; margin-bottom: 24px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4); display: flex; width: 100%; }
    div[data-testid="stSegmentedControl"] button { font-weight: 700 !important; font-size: 13px !important; letter-spacing: 0.5px !important; color: #94a3b8 !important; border-radius: 8px !important; padding: 8px 16px !important; flex: 1; transition: all 0.25s ease !important; }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] { background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important; color: #ffffff !important; box-shadow: 0 0 15px rgba(56, 189, 248, 0.4) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar ---
st.sidebar.markdown("<h2 style='font-weight: 800; letter-spacing: -0.5px;'><i class='fa-solid fa-satellite-dish'></i> IBVAP Link</h2>", unsafe_allow_html=True)
backend_url = st.sidebar.text_input("Central API URL", value="http://127.0.0.1:8000")
refresh_rate = st.sidebar.slider("Auto-Refresh Interval (s)", min_value=1, max_value=15, value=3)
auto_refresh = st.sidebar.checkbox("Enable Live Auto-Refresh", value=True)
st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='font-weight: 700; font-size: 16px;'><i class='fa-solid fa-filter'></i> Tactical Filters</h3>", unsafe_allow_html=True)
priority_filter = st.sidebar.selectbox("Priority Level", ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
status_filter = st.sidebar.selectbox("Operator Review", ["ALL", "PENDING_REVIEW", "CONFIRMED_BREACH", "FALSE_ALARM"])
camera_filter = st.sidebar.text_input("Camera Designation (e.g. CAM-01)", value="")
st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='font-weight: 700; font-size: 14px;'><i class='fa-solid fa-users'></i> Development Team</h3>", unsafe_allow_html=True)
team_html = """
<div style="font-size: 12px; line-height: 2.0;">
    <i class="fa-solid fa-envelope" style="color: #94a3b8; margin-right: 6px;"></i><a href="mailto:23f3003672@ds.study.iitm.ac.in" style="color: #38bdf8; text-decoration: none; font-weight: 500;">Devansh</a><br>
    <i class="fa-solid fa-envelope" style="color: #94a3b8; margin-right: 6px;"></i><a href="mailto:nitishswaggaming@gmail.com" style="color: #38bdf8; text-decoration: none; font-weight: 500;">Nitish</a><br>
    <i class="fa-solid fa-envelope" style="color: #94a3b8; margin-right: 6px;"></i><a href="mailto:pt286355@gmail.com" style="color: #38bdf8; text-decoration: none; font-weight: 500;">Prashant</a><br>
    <i class="fa-solid fa-envelope" style="color: #94a3b8; margin-right: 6px;"></i><a href="mailto:bhumikashaw.549@gmail.com" style="color: #38bdf8; text-decoration: none; font-weight: 500;">Bhumika</a><br>
    <i class="fa-solid fa-envelope" style="color: #94a3b8; margin-right: 6px;"></i><a href="mailto:24f3001643@ds.study.iitm.ac.in" style="color: #38bdf8; text-decoration: none; font-weight: 500;">Arnav</a><br>
    <i class="fa-solid fa-envelope" style="color: #94a3b8; margin-right: 6px;"></i><a href="mailto:25f1001964@ds.study.iitm.ac.in" style="color: #38bdf8; text-decoration: none; font-weight: 500;">Garima</a>
</div>
"""
st.sidebar.markdown(team_html, unsafe_allow_html=True)


# --- Data Fetching ---
def fetch_backend_health(url: str) -> bool:
    try:
        r = requests.get(f"{url}/api/v1/health", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False

def fetch_stats(url: str) -> Dict:
    try:
        r = requests.get(f"{url}/api/v1/stats", timeout=1.5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}

def fetch_alerts(url: str, priority: str, status_f: str, cam_f: str) -> List[Dict]:
    params = {"limit": 100}
    if priority != "ALL":
        params["priority"] = priority
    if status_f != "ALL":
        params["feedback_status"] = status_f
    if cam_f:
        params["camera_id"] = cam_f
    try:
        r = requests.get(f"{url}/api/v1/alerts", params=params, timeout=2.5)
        if r.status_code == 200:
            return r.json().get("alerts", [])
    except Exception as e:
        st.error(f"Error connecting to backend API: {e}")
    return []

def fetch_cameras(url: str) -> List[Dict]:
    """Phase 5: Fetches active camera list from backend."""
    try:
        r = requests.get(f"{url}/api/v1/cameras", timeout=2.0)
        if r.status_code == 200:
            return r.json().get("cameras", [])
    except Exception:
        pass
    return []

def fetch_zone(url: str, camera_id: str) -> List:
    """Phase 5: Fetches existing zone polygon for a camera."""
    try:
        r = requests.get(f"{url}/api/v1/cameras/{camera_id}/zones", timeout=2.0)
        if r.status_code == 200:
            return r.json().get("polygon", [])
    except Exception:
        pass
    return []

def submit_feedback(url: str, alert_id: str, action: str, notes: str):
    try:
        payload = {"action": action, "operator_id": "OPERATOR-HQ-01", "notes": notes}
        r = requests.post(f"{url}/api/v1/alerts/{alert_id}/feedback", json=payload, timeout=2.0)
        if r.status_code == 200:
            st.toast(f"Feedback logged: {action} on {alert_id}", icon="\u2714\ufe0f")
            time.sleep(0.3)
            st.rerun()
    except Exception as e:
        st.error(f"Failed to submit feedback: {e}")

def render_alert_card(alert: Dict, url: str, key_prefix: str = ""):
    """Renders a single alert card with telemetry and feedback buttons."""
    alert_id = alert["alert_id"]
    priority_level = alert.get("priority_level", "LOW")
    priority_score = alert.get("priority_score", 0.0)
    primary_rule = alert.get("primary_rule", "Nominal Detection")
    obj_class = alert.get("object_class", "Unidentified Target").upper()
    coords = alert.get("world_coords", {})
    vel = alert.get("velocity_mps", 0.0)
    zone = alert.get("zone", "Unspecified Zone").replace("_", " ")
    plate = alert.get("license_plate")
    suspect = alert.get("suspect_id")
    face_conf = alert.get("face_confidence", 0.0)
    timestamp_str = alert.get("timestamp", "").replace("T", " ").replace("Z", "")
    thumb = alert.get("thumbnail_b64")

    badge_cls = f"badge-{priority_level.lower()}"
    card_cls = f"card-{priority_level.lower()}" if not suspect else "card-critical"

    suspect_badge = '<span class="badge badge-critical" style="margin-left: 6px;"><i class="fa-solid fa-user-ninja"></i> WATCHLIST MATCH</span>' if suspect else ""

    plate_html = (
        f'<div class="telemetry-item" style="grid-column: span 2;">'
        f'<div class="telemetry-lbl">ANPR RECOGNITION</div>'
        f'<div class="telemetry-val" style="color: #fef08a; font-size: 14px; letter-spacing: 2px; text-align: center;">'
        f'<i class="fa-solid fa-car-side"></i> {plate}</div></div>'
    ) if plate else ""

    suspect_html = (
        f'<div class="telemetry-item" style="grid-column: span 2; background: rgba(127, 29, 29, 0.45); border: 1px solid #ef4444;">'
        f'<div class="telemetry-lbl" style="color: #fca5a5;"><i class="fa-solid fa-triangle-exclamation"></i> BIOMETRIC WATCHLIST HIT</div>'
        f'<div class="telemetry-val" style="color: #fee2e2; font-size: 13px; font-weight: 700; letter-spacing: 0.5px;">'
        f'{suspect} ({int(face_conf * 100) if face_conf else 0}% match)</div></div>'
    ) if suspect else ""

    thumb_html = f'<img src="{thumb}" class="compressed-img" style="max-width:380px; margin-bottom: 12px;"/>' if (thumb and thumb.startswith("data:image")) else """<div style="background: rgba(0,0,0,0.4); height: 120px; display: flex; align-items: center;
                justify-content: center; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);
                color: #475569; font-size: 12px; font-weight: 500; margin-bottom: 12px;">
                <i class="fa-solid fa-image-slash" style="margin-right: 8px;"></i> No Visual Evidence</div>"""

    card_html = f"""
    <div class="alert-card {card_cls}">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
            <div>
                <span class="badge {badge_cls}">{priority_level} : SCORE {priority_score:.1f}</span>
                {suspect_badge}
                <h3 style="margin: 8px 0 0 0; font-size: 16px; font-weight: 700;">{primary_rule}</h3>
            </div>
            <div style="text-align: right; font-size: 11px;">
                <div style="color: #cbd5e1;"><i class="fa-regular fa-clock"></i> {timestamp_str.split(' ')[1] if ' ' in timestamp_str else ''} UTC</div>
            </div>
        </div>
        <div style="color: #94a3b8; font-size: 11px; margin-bottom: 12px;">
            <i class="fa-solid fa-camera"></i> {alert.get('camera_id', 'CAM-01')} | {zone}
        </div>
        {thumb_html}
        <div class="telemetry-grid">
            <div class="telemetry-item"><div class="telemetry-lbl">Target</div><div class="telemetry-val" style="color: #38bdf8;">{obj_class}</div></div>
            <div class="telemetry-item"><div class="telemetry-lbl">Speed</div><div class="telemetry-val">{vel:.1f} m/s</div></div>
            {plate_html}
            {suspect_html}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
    if thumb and thumb.startswith("data:image"):
        with st.expander("🔍 View Full Evidence"):
            st.image(thumb, width="stretch")

    btn1, btn2 = st.columns(2)
    with btn1:
        if st.button("Dispatch", key=f"{key_prefix}conf_{alert_id}", help="Confirm threat and dispatch units"):
            submit_feedback(url, alert_id, "CONFIRMED_BREACH", "Threat confirmed.")
    with btn2:
        if st.button("Mark False", key=f"{key_prefix}false_{alert_id}", help="Flag as false alarm for retraining"):
            submit_feedback(url, alert_id, "FALSE_ALARM", "Flagged.")
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


# --- Setup Wizard / System Configuration ---
def fetch_config_status(url):
    try:
        r = requests.get(f"{url}/api/v1/config/status", timeout=2.0)
        if r.status_code == 200:
            return r.json().get("is_configured", False)
    except Exception:
        pass
    # If backend is down or not responding properly, assume True so we can show the Offline KPI later
    return True

is_configured = fetch_config_status(backend_url)

if not is_configured:
    st.markdown(
        """
        <div style="text-align: center; margin-top: 50px;">
            <h1 style="font-size: 32px; font-weight: 800; color: #f8fafc;">
                <i class="fa-solid fa-shield-halved" style="color: #38bdf8; margin-right: 12px;"></i>
                SYSTEM INITIALIZATION
            </h1>
            <p style="color: #94a3b8; font-size: 16px;">The Intelligent Border Video Analytics Platform requires initial configuration.</p>
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 30px 0;'>", unsafe_allow_html=True)
    st.markdown("### <i class='fa-solid fa-camera'></i> Camera Surveillance Zones", unsafe_allow_html=True)
    
    cameras = fetch_cameras(backend_url)
    if not cameras:
        st.info("Waiting for edge nodes and cameras to connect...")
        time.sleep(2)
        st.rerun()
        
    config_state = {}
    with st.form("initial_config_form"):
        st.markdown("<p style='color: #cbd5e1; font-size: 14px; margin-bottom: 20px;'>Assign a surveillance mode to each connected camera.</p>", unsafe_allow_html=True)
        
        cols = st.columns(3)
        for i, cam_dict in enumerate(cameras):
            cam_id = cam_dict.get("camera_id", f"CAM-{i}")
            with cols[i % 3]:
                # Default to 'Civilian zone' (index 1) as requested
                mode = st.selectbox(
                    f"{cam_id}",
                    ["Alert zone", "Civilian zone", "No Civilian zone", "No vehicle zone", "Emergency/sensitive zone"],
                    index=1,
                    key=f"init_{cam_id}"
                )
                config_state[cam_id] = mode
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Save Configuration & Start System", type="primary", use_container_width=True)
        
        if submitted:
            try:
                r = requests.post(f"{backend_url}/api/v1/config/init", json={"camera_zones": config_state}, timeout=5.0)
                if r.status_code == 200:
                    st.success("Configuration saved! Booting detection pipelines...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Failed to save configuration: {r.text}")
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")
                
    st.stop()


# --- Main UI ---
is_online = fetch_backend_health(backend_url)
stats = fetch_stats(backend_url)

# Header
st.markdown(
    """
    <div class="header-container">
        <div>
            <h1 class="header-title"><i class="fa-solid fa-layer-group" style="margin-right: 10px;"></i>Intelligent Border Video Analytics Platform</h1>
            <div class="header-subtitle">Tactical Threat Command Center (SIH26187) &bull; Developed for SSB</div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 11px; color: #94a3b8; font-weight: 600; letter-spacing: 1px; text-transform: uppercase;">System Time (UTC)</div>
            <div style="font-size: 20px; font-weight: 700; color: #f8fafc; font-family: monospace;">{time}</div>
        </div>
    </div>
    """.replace("{time}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    unsafe_allow_html=True,
)

# KPI Strip
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    color = "#10b981" if is_online else "#ef4444"
    text = "ONLINE" if is_online else "OFFLINE"
    icon = "fa-wifi" if is_online else "fa-network-wired"
    st.markdown(f'<div class="kpi-card"><div class="kpi-lbl"><i class="fa-solid {icon}"></i> API Link Status</div><div class="kpi-val" style="color: {color}; font-size: 28px; padding-top: 6px;">{text}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-lbl"><i class="fa-solid fa-eye"></i> Total Detections</div><div class="kpi-val" style="color: #38bdf8;">{stats.get("total_alerts", 0)}</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-lbl"><i class="fa-solid fa-triangle-exclamation"></i> Critical Threats</div><div class="kpi-val" style="color: #ef4444;">{stats.get("priority_breakdown", {}).get("CRITICAL", 0)}</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-lbl"><i class="fa-solid fa-bolt"></i> High Priority</div><div class="kpi-val" style="color: #f97316;">{stats.get("priority_breakdown", {}).get("HIGH", 0)}</div></div>', unsafe_allow_html=True)
with col5:
    st.markdown(f'<div class="kpi-card"><div class="kpi-lbl"><i class="fa-solid fa-clipboard-list"></i> Pending Review</div><div class="kpi-val" style="color: #eab308;">{stats.get("review_breakdown", {}).get("PENDING_REVIEW", 0)}</div></div>', unsafe_allow_html=True)

st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TACTICAL NAVIGATION TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_options = [
    "📹  Live Feeds",
    "🗺️  Tactical Map",
    "🛡️  Alert Queue",
    "⚙️  Camera Config",
    "🤖  AI Reports",
]
selected_tab = st.segmented_control(
    "Tactical Sector Navigation",
    tab_options,
    default="📹  Live Feeds",
    key="active_tactical_tab",
    label_visibility="collapsed",
)
if not selected_tab:
    selected_tab = tab_options[0]


# ─── TAB 1: LIVE FEEDS ───────────────────────────────────────────────────────
if selected_tab == tab_options[0]:
    st.markdown(
        "<h2 style='font-size: 22px; font-weight: 700; margin-bottom: 4px;'>"
        "<i class='fa-solid fa-video'></i> Live Surveillance Wall</h2>"
        "<p style='color: #94a3b8; font-size: 13px; margin-bottom: 20px;'>"
        "AI-processed MJPEG feeds &bull; Zero controls &bull; Continuous loop &bull; Detection-gated pipeline active</p>",
        unsafe_allow_html=True,
    )

    grid_size_option = st.selectbox("Grid Layout", ["1x1", "2x2", "3x3", "4x4", "5x5", "10x10"], index=1)
    COLS = int(grid_size_option.split("x")[0])

    cameras = fetch_cameras(backend_url)

    if not cameras:
        st.markdown(
            """<div style="background: rgba(15,23,42,0.5); border: 1px dashed rgba(56,189,248,0.25);
                    border-radius: 14px; padding: 48px; text-align: center; color: #475569;">
                <div style="font-size: 48px; margin-bottom: 16px;"><i class="fa-solid fa-camera-slash"></i></div>
                <div style="font-size: 16px; font-weight: 700; color: #64748b;">No Active Camera Streams</div>
                <div style="font-size: 13px; margin-top: 8px; color: #475569;">
                    Ensure the backend is running and <code>sample-videos/</code> contains landscape MP4 files.
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        rows = [cameras[i:i + COLS] for i in range(0, len(cameras), COLS)]
        for row in rows:
            grid_cols = st.columns(COLS, gap="small")
            for col_idx, cam in enumerate(row):
                cam_id = cam["camera_id"]
                source_name = cam.get("source", "")
                stream_url = f"{backend_url}/api/v1/stream/{cam_id}"
                with grid_cols[col_idx]:
                    st.markdown(
                        f"""<div class="cctv-tile-wrapper">
                            <img src="{stream_url}"
                                 alt="Live feed {cam_id}"
                                 id="cctv-{cam_id.lower().replace('-','')}"
                                 draggable="false"
                                 title="{cam_id}">
                            <div class="cctv-label">
                                <i class="fa-solid fa-circle-dot" style="color: #22c55e; margin-right: 5px;"></i>{cam_id}
                            </div>
                            <div class="cctv-live-dot"></div>
                        </div>
                        <div style="font-size: 10px; color: #475569; text-align: center; margin-top: 5px; letter-spacing: 0.5px;">{source_name}</div>""",
                        unsafe_allow_html=True,
                    )
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)


# ─── TAB 2: TACTICAL MAP ─────────────────────────────────────────────────────
elif selected_tab == tab_options[1]:
    alerts = fetch_alerts(backend_url, priority_filter, status_filter, camera_filter)
    col_map, col_queue = st.columns([7, 3])

    with col_map:
        st.markdown("<h2 style='font-size: 22px; font-weight: 700; margin-bottom: 2px;'><i class='fa-solid fa-map-location-dot'></i> Tactical Geospatial View</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8; font-size: 14px; margin-bottom: 16px;'>Real-time plotting of incursions using edge homography calculations.</p>", unsafe_allow_html=True)

        center_lat, center_lon = 28.6139, 77.2090
        valid_coords = []
        for a in alerts:
            coords = a.get("world_coords", {})
            if coords and coords.get("x", 0.0) != 0.0:
                valid_coords.append((coords["x"], coords["y"]))
        if valid_coords:
            center_lat = sum([c[0] for c in valid_coords]) / len(valid_coords)
            center_lon = sum([c[1] for c in valid_coords]) / len(valid_coords)

        carto_api_key = "cb1_2kbh_1_6816a26dfec7dcc8c58f9cce"
        tile_url = f"https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}.png?key={carto_api_key}"
        attr = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, &copy; <a href="https://carto.com/attributions">CARTO</a>'

        m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles=tile_url, attr=attr)

        for alert in alerts:
            coords = alert.get("world_coords", {})
            lat, lon = coords.get("x", 0.0), coords.get("y", 0.0)
            if lat != 0.0 and lon != 0.0:
                obj_class = alert.get("object_class", "Unidentified Object").lower()
                if obj_class == "person":
                    color, icon = "red", "person"
                elif obj_class in ["vehicle", "car", "truck"]:
                    color, icon = "blue", "car"
                else:
                    color, icon = "orange", "circle-info"
                priority = alert.get("priority_level", "LOW")
                folium.Marker(
                    [lat, lon],
                    popup=folium.Popup(f"<b>{priority} Threat</b><br>Type: {obj_class.title()}<br>Speed: {alert.get('velocity_mps', 0)} m/s", max_width=200),
                    tooltip=f"ID: {alert['alert_id']}",
                    icon=folium.Icon(color=color, icon=icon, prefix='fa')
                ).add_to(m)

        st_folium(m, width="100%", height=700, use_container_width=True, returned_objects=[])

    with col_queue:
        st.markdown(f"<h2 style='font-size: 22px; font-weight: 700; margin-bottom: 2px;'><i class='fa-solid fa-shield-halved'></i> Ranked Queue ({len(alerts)})</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8; font-size: 14px; margin-bottom: 16px;'>Priority cognitive engine.</p>", unsafe_allow_html=True)
        if not alerts:
            st.markdown(
                """<div style="background: rgba(15, 23, 42, 0.4); border: 1px dashed rgba(255,255,255,0.1); border-radius: 12px; padding: 30px; text-align: center; color: #64748b;">
                    <div style="font-size: 32px; margin-bottom: 12px;"><i class="fa-solid fa-shield-check"></i></div>
                    <div style="font-weight: 600; font-size: 14px;">Sector Secure.</div></div>""",
                unsafe_allow_html=True
            )
        else:
            with st.container(height=700, border=True):
                for alert in alerts:
                    render_alert_card(alert, backend_url, key_prefix="map_")


# ─── TAB 3: ALERT QUEUE ──────────────────────────────────────────────────────
elif selected_tab == tab_options[2]:
    alerts = fetch_alerts(backend_url, priority_filter, status_filter, camera_filter)
    st.markdown(
        f"<h2 style='font-size: 22px; font-weight: 700; margin-bottom: 2px;'>"
        f"<i class='fa-solid fa-shield-halved'></i> Ranked Alert Queue "
        f"<span style='color:#94a3b8; font-size:16px;'>({len(alerts)} events)</span></h2>"
        f"<p style='color: #94a3b8; font-size: 13px; margin-bottom: 20px;'>"
        f"Sorted by Priority Score \u2193 \u2022 Critical \u2192 High \u2192 Medium \u2192 Low</p>",
        unsafe_allow_html=True,
    )
    if not alerts:
        st.markdown(
            """<div style="background: rgba(15,23,42,0.4); border: 1px dashed rgba(255,255,255,0.1); border-radius: 12px; padding: 48px; text-align: center; color: #64748b;">
                <div style="font-size: 32px; margin-bottom: 12px;"><i class="fa-solid fa-shield-check"></i></div>
                <div style="font-weight: 600; font-size: 15px;">Sector Secure \u2014 No Active Alerts</div></div>""",
            unsafe_allow_html=True,
        )
    else:
        q_col1, q_col2 = st.columns(2, gap="medium")
        mid = (len(alerts) + 1) // 2
        for idx, alert in enumerate(alerts):
            with (q_col1 if idx < mid else q_col2):
                render_alert_card(alert, backend_url, key_prefix="queue_")


# ─── TAB 4: CAMERA CONFIG ────────────────────────────────────────────────────
elif selected_tab == tab_options[3]:
    st.markdown(
        "<h2 style='font-size: 22px; font-weight: 700; margin-bottom: 4px;'>"
        "<i class='fa-solid fa-camera'></i> Camera Operational Mode</h2>"
        "<p style='color: #94a3b8; font-size: 13px; margin-bottom: 20px;'>"
        "Set the camera-wide surveillance mode. "
        "Detected objects across the entire frame will be evaluated according to this mode.</p>",
        unsafe_allow_html=True,
    )

    cameras = fetch_cameras(backend_url)

    if not cameras:
        st.warning("\u26a0\ufe0f No active cameras found. Start the backend with sample videos to configure zones.")
    else:
        cam_ids = [c["camera_id"] for c in cameras]
        selected_cam = st.selectbox(
            "Select Camera to Configure",
            cam_ids,
            key="zone_cam_select",
            help="Choose the camera feed you wish to assign a restricted zone to.",
        )

        surveillance_mode = st.selectbox(
            "Surveillance Mode",
            ["Alert zone", "Civilian zone", "No Civilian zone", "No vehicle zone", "Emergency/sensitive zone"],
            help="Configure what triggers a zone intrusion alert."
        )

        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 16px 0;'>", unsafe_allow_html=True)

        if st.button("\U0001f4be Save Configuration", key=f"save_config_{selected_cam}", width="stretch"):
            try:
                resp = requests.post(
                    f"{backend_url}/api/v1/cameras/{selected_cam}/zones",
                    json={"polygon": [], "zone_label": surveillance_mode},
                    timeout=3.0,
                )
                if resp.status_code == 200:
                    st.success(f"✅ Surveillance mode '{surveillance_mode}' saved for **{selected_cam}**. Engine updated immediately.")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Backend error: {resp.text}")
            except Exception as e:
                st.error(f"Failed to save config: {e}")



# ─── TAB 5: AI REPORTS ───────────────────────────────────────────────────────
elif selected_tab == tab_options[4]:
    st.markdown(
        "<h2 style='font-size: 22px; font-weight: 700; margin-bottom: 4px;'>"
        "<i class='fa-solid fa-robot'></i> AI Copilot & Reports</h2>"
        "<p style='color: #94a3b8; font-size: 13px; margin-bottom: 20px;'>"
        "Natural language querying of historical border incursions using Local LLM & RAG.</p>",
        unsafe_allow_html=True,
    )

    query = st.text_input("Ask a question about recent alerts (e.g., 'Summarize incursions on CAM-BOP-01'):", value="Give me a daily incident report for CAM-BOP-01.")

    if st.button("Generate AI Report", type="primary", width="stretch"):
        with st.spinner("Retrieving historical context and analyzing incursions..."):
            try:
                # 1. Ask Backend for Context & Synthesized Report
                r = requests.post(f"{backend_url}/api/v1/investigate", json={"query": query}, timeout=8.0)
                if r.status_code == 200:
                    data = r.json()
                    generated_prompt = data.get("generated_prompt", "")
                    synthesized_report = data.get("synthesized_report", "")

                    report_markdown = ""
                    source_badge = ""

                    # 2. Try Local Ollama if available
                    try:
                        ollama_url = "http://localhost:11434/api/generate"
                        ollama_payload = {
                            "model": "llama3",
                            "prompt": generated_prompt,
                            "stream": False
                        }
                        ollama_r = requests.post(ollama_url, json=ollama_payload, timeout=2.0)
                        if ollama_r.status_code == 200:
                            report_markdown = ollama_r.json().get("response", "")
                            source_badge = "🤖 Generated by Local LLM (Ollama llama3)"
                    except Exception:
                        pass # Seamless fallback to tactical RAG synthesizer

                    # 3. If Ollama is offline or unavailable, use the Tactical RAG Intelligence Engine
                    if not report_markdown:
                        report_markdown = synthesized_report or "No incident telemetry found matching query."
                        source_badge = "⚡ Generated by Built-in Tactical RAG Intelligence Engine (Ollama offline at localhost:11434)"

                    st.markdown(
                        f"<div style='display: inline-block; background: rgba(56, 189, 248, 0.12); "
                        f"border: 1px solid rgba(56, 189, 248, 0.35); border-radius: 6px; "
                        f"padding: 5px 14px; font-size: 11px; font-weight: 700; color: #38bdf8; margin-bottom: 16px;'>"
                        f"{source_badge}</div>",
                        unsafe_allow_html=True
                    )

                    with st.container(border=True):
                        st.markdown(report_markdown)

                    with st.expander("🔍 View Raw RAG Context & LLM Prompt Template", expanded=False):
                        st.code(generated_prompt, language="markdown")

                else:
                    st.error(f"Backend RAG API error: {r.text}")
            except Exception as e:
                st.error(f"Error querying backend: {e}")
    else:
        st.markdown(
            """<div style="background: rgba(15,23,42,0.4); border: 1px dashed rgba(255,255,255,0.1); border-radius: 12px; padding: 48px; text-align: center; color: #64748b; margin-top: 20px;">
                <div style="font-size: 32px; margin-bottom: 12px;"><i class="fa-solid fa-file-invoice"></i></div>
                <div style="font-weight: 600; font-size: 15px;">Report generator ready. Enter a query and click generate.</div></div>""",
            unsafe_allow_html=True,
        )

# --- Auto Refresh ---
# Only refresh when on monitoring tabs so configuration & AI prompt typing are never interrupted
if auto_refresh and selected_tab in [tab_options[0], tab_options[1], tab_options[2]]:
    time.sleep(refresh_rate)
    st.rerun()
