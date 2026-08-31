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
