from functools import lru_cache
from typing import List, Any

from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------
# LAZY MODEL LOADING
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedding_model():

    print(
        f"[GenAI] Loading embedding model: {MODEL_NAME}"
    )

    model = SentenceTransformer(MODEL_NAME)

    print(
        "[GenAI] Embedding model loaded successfully."
    )

    return model


# ---------------------------------------------------------
# ALERT → SEMANTIC TEXT
# ---------------------------------------------------------

def translate_alert_to_text(alert: dict) -> str:

    object_class = alert.get(
        "object_class",
        "unknown object",
    )

    coords = alert.get(
        "world_coords",
        {},
    )

    x = float(
        coords.get("x", 0.0)
    )

    y = float(
        coords.get("y", 0.0)
    )

    timestamp = alert.get(
        "timestamp",
        "unknown time",
    )

    velocity = float(
        alert.get(
            "velocity_mps",
            0.0,
        )
    )

    heading = float(
        alert.get(
            "heading_deg",
            0.0,
        )
    )

    priority = alert.get(
        "priority_level",
        "UNKNOWN",
    )

    camera = alert.get(
        "camera_id",
        "unknown camera",
    )

    sentence = (
        f"A {priority} priority {object_class} "
        f"was detected by camera {camera} "
        f"at coordinates {x:.5f}, {y:.5f} "
        f"on {timestamp}, "
        f"moving at {velocity:.1f} m/s "
        f"with heading {heading:.1f} degrees."
    )

    feedback = alert.get(
        "feedback_status"
    )

    if feedback == "CONFIRMED_BREACH":

        sentence += (
            " The operator confirmed this event "
            "as a critical breach."
        )

    elif feedback == "FALSE_ALARM":

        sentence += (
            " The operator marked this event "
            "as a false alarm."
        )

    elif alert.get("is_threat"):

        sentence += (
            " The AI classified this event "
            "as a potential threat."
        )

    return sentence


# ---------------------------------------------------------
# TEXT → VECTOR
# ---------------------------------------------------------

def embed_text(text: str) -> List[float]:

    if not text or not text.strip():

        raise ValueError(
            "Cannot generate embedding from empty text."
        )

    model = get_embedding_model()

    embedding = model.encode(
        text,
        normalize_embeddings=True,
    )

    return embedding.tolist()


# ---------------------------------------------------------
# RAG PROMPT
# ---------------------------------------------------------

def generate_rag_prompt(
    query: str,
    search_results: List[Any],
) -> str:

    if not search_results:

        context_text = (
            "No relevant historical alerts "
            "were retrieved."
        )

    else:

        context_blocks = []

        for index, result in enumerate(
            search_results,
            start=1,
        ):

            context_blocks.append(
                f"[{index}] "
                f"Alert ID: {result.alert_id}\n"
                f"{result.semantic_text}"
            )

        context_text = "\n\n".join(
            context_blocks
        )

    prompt = f"""
You are an AI tactical assistant for the
IBVAP Intelligent Border Video Analytics Platform.

Answer the user's question using ONLY the
retrieved alert context.

Rules:
- Do not invent information.
- If the answer is not present in the context,
  clearly say that the information is unavailable.
- Keep the answer concise and factual.

--- RETRIEVED ALERT CONTEXT ---

{context_text}

--- USER QUERY ---

{query}

--- ANSWER ---
"""

    return prompt.strip()