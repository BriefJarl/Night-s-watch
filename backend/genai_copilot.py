import os
from sentence_transformers import SentenceTransformer

# Initialize the embedding model globally so it stays in memory
# Utilizing HuggingFace's all-MiniLM-L6-v2 as requested for dense vector generation
model_name = "sentence-transformers/all-MiniLM-L6-v2"
try:
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    print("Embedding model loaded successfully.")
except Exception as e:
    print(f"Warning: Failed to load SentenceTransformer model. Error: {e}")
    model = None

def translate_alert_to_text(alert: dict) -> str:
    """
    Translates a JSON alert payload into a semantic natural language sentence.
    
    Example output: 
    'A vehicle was detected at coordinates 28.1, 77.2 on 2026-08-28T23:45:00Z moving at 15.5 m/s with a heading of 45.0 degrees.'
    """
    object_class = alert.get("object_class", "unknown object")
    
    # Extract coordinates
    coords = alert.get("world_coords", {})
    x = coords.get("x", 0.0)
    y = coords.get("y", 0.0)
    
    timestamp = alert.get("timestamp", "unknown time")
    velocity = alert.get("velocity_mps", 0.0)
    heading = alert.get("heading_deg", 0.0)
    priority = alert.get("priority_level", "UNKNOWN")
    camera = alert.get("camera_id", "unknown camera")

    sentence = (
        f"A {priority} priority {object_class} was detected by {camera} "
        f"at coordinates {x:.5f}, {y:.5f} on {timestamp} "
        f"moving at {velocity:.1f} m/s with a heading of {heading:.1f} degrees."
    )

    if alert.get("suspect_id"):
        conf_pct = int(alert.get("face_confidence", 0.0) * 100)
        sentence += f" CRITICAL BIOMETRIC HIT: Watchlist suspect [{alert.get('suspect_id')}] was positively identified with {conf_pct}% confidence."

    if alert.get("license_plate"):
        sentence += f" The vehicle was verified with license plate [{alert.get('license_plate')}]."
    
    feedback = alert.get("feedback_status")
    if feedback == "CONFIRMED_BREACH":
        sentence += " This was manually CONFIRMED as a critical breach by the operator."
    elif feedback == "FALSE_ALARM":
        sentence += " This was flagged as a FALSE ALARM by the operator."
    elif alert.get("is_threat"):
        sentence += " It was flagged as a potential threat by the AI."
        
    return sentence


def embed_text(text: str) -> list[float]:
    """
    Converts a natural language string into a 384-dimensional vector array.
    """
    if model is None:
        # Fallback to zero vector if model failed to load
        print("Warning: Model not loaded, returning zero vector.")
        return [0.0] * 384
        
    # Generate embedding
    embedding = model.encode(text)
    
    # Return as a simple list of floats for pgvector insertion
    return embedding.tolist()

def generate_rag_prompt(query: str, search_results: list) -> str:
    """
    Constructs an LLM prompt template using the user's query and the retrieved contexts.
    
    If you integrate an LLM (e.g., via openai package or ollama), you would pass this 
    prompt to the LLM to get the final response.
    """
    context_blocks = []
    for i, result in enumerate(search_results, 1):
        context_blocks.append(f"[{i}] {result.semantic_text}")
        
    context_text = "\n".join(context_blocks)
    
    prompt = f"""You are a tactical assistant for the IBVAP system.
Based on the following historical alerts retrieved from the database, answer the officer's query.
Keep your answer concise, factual, and strictly based on the provided context.

--- Context ---
{context_text}

--- Query ---
{query}

--- Response ---
"""
    return prompt


def synthesize_incident_report(query: str, context_docs: list) -> str:
    """
    Synthesizes an executive tactical incident report directly from retrieved context alerts.
    Provides instant, deterministic tactical intelligence when external LLMs (Ollama) are offline.
    """
    import re
    from collections import Counter

    if not context_docs:
        return "No relevant historical alert events found matching this sector query."

    total_events = len(context_docs)
    priorities = []
    targets = []
    cameras = set()
    suspects = []
    plates = []
    max_velocity = 0.0
    timestamps = []

    for doc in context_docs:
        text = doc.get("text", "") if isinstance(doc, dict) else str(doc)

        # Priority
        pri_match = re.search(r"\b(CRITICAL|HIGH|MEDIUM|LOW)\b", text)
        if pri_match:
            priorities.append(pri_match.group(1))

        # Target class
        for cls in ["person", "car", "vehicle", "truck", "motorcycle", "bicycle"]:
            if re.search(rf"\b{cls}\b", text, re.IGNORECASE):
                targets.append(cls.title())
                break

        # Camera
        cam_match = re.search(r"\b(CAM-[A-Z0-9_-]+)\b", text)
        if cam_match:
            cameras.add(cam_match.group(1))

        # Suspect hit
        suspect_match = re.search(r"Watchlist suspect \[([^\]]+)\](?:\s+was positively identified with (\d+)% confidence)?", text)
        if suspect_match:
            name = suspect_match.group(1)
            conf = suspect_match.group(2)
            suspects.append(f"**{name}** ({conf}% match)" if conf else f"**{name}**")

        # License plate
        plate_match = re.search(r"license plate \[([^\]]+)\]", text)
        if plate_match:
            plates.append(plate_match.group(1))

        # Velocity
        vel_match = re.search(r"moving at ([0-9.]+)\s*m/s", text)
        if vel_match:
            try:
                v = float(vel_match.group(1))
                if v > max_velocity:
                    max_velocity = v
            except ValueError:
                pass

        # Timestamp
        time_match = re.search(r"on\s+([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z?)", text)
        if time_match:
            timestamps.append(time_match.group(1))

    # Aggregate summaries
    pri_counts = Counter(priorities)
    target_counts = Counter(targets)
    cameras_str = ", ".join(sorted(cameras)) if cameras else "Monitored Border Sector"
    
    highest_pri = "CRITICAL" if "CRITICAL" in pri_counts else ("HIGH" if "HIGH" in pri_counts else "MEDIUM")
    status_icon = "🔴" if highest_pri == "CRITICAL" else ("🟡" if highest_pri == "HIGH" else "🔵")

    pri_summary = ", ".join([f"{cnt} {lvl}" for lvl, cnt in pri_counts.items()]) or f"{total_events} events"
    target_summary = ", ".join([f"{cnt} {t}" for t, cnt in target_counts.items()]) or "Multiple unidentified objects"

    time_range_str = f"From `{timestamps[0][:19].replace('T', ' ')} UTC` to `{timestamps[-1][:19].replace('T', ' ')} UTC`" if timestamps else "Recent surveillance window"

    suspect_section = ""
    if suspects:
        unique_suspects = list(dict.fromkeys(suspects))
        suspect_section = f"\n- 🚨 **Watchlist Biometric Hit:** High-priority match identified: {', '.join(unique_suspects)}."
    
    plate_section = ""
    if plates:
        unique_plates = list(dict.fromkeys(plates))
        plate_section = f"\n- 🚘 **Verified Plate(s):** ANPR detected plates: {', '.join([f'`{p}`' for p in unique_plates])}."

    recommendation = (
        "Immediate perimeter lockdown advised. Dispatch Quick Reaction Team (QRT) to sector grid coordinates. Intercept high-value watchlist target."
        if suspects else
        "Maintain heightened surveillance posture. Direct PTZ cameras toward ingress vector and monitor telemetry."
    )

    report = (
        f"### 🛡️ Tactical Incident Intelligence Report\n\n"
        f"**Target Sector:** `{cameras_str}` &nbsp;|&nbsp; **Classification:** {status_icon} **{highest_pri} PRIORITY BREACH**\n\n"
        f"**Observation Period:** {time_range_str}\n\n"
        f"#### 1. Threat Summary & Telemetry\n"
        f"- **Total Indexed Events:** {total_events} security detections ({pri_summary})\n"
        f"- **Detected Ingress Entities:** {target_summary}\n"
        f"- **Maximum Observed Velocity:** `{max_velocity:.1f} m/s` (Dynamic Vector Tracking)"
        f"{suspect_section}"
        f"{plate_section}\n\n"
        f"#### 2. Pattern Analysis\n"
        f"Multiple boundary breach indicators logged at {cameras_str}. Movement vectors demonstrate sustained spatial presence "
        f"consistent with perimeter scouting or unauthorized border transit.\n\n"
        f"#### 3. Recommended Tactical Action\n"
        f"> **ACTION DIRECTIVE:** {recommendation}\n"
    )
    return report

