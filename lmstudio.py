import streamlit as st
import requests
import json
from urllib.parse import urljoin, urlparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Optional, Dict, Any, Iterator 

# --- App Configuration ---
LM_STUDIO_DEFAULT_BASE_URL = "http://localhost:1234/v1" 
QWEN3_CHAT_MODEL_ID = "qwen3-14b-128K-Q4_K_M.gguf" 
NOMIC_EMBEDDING_MODEL_ID = "nomic-embed-text-v1.5.Q8_0.gguf"

DEFAULT_SYSTEM_PROMPT_RAG = """You are a highly intelligent AI assistant.
The user is asking a question. To help you answer, relevant excerpts from your previous conversation with the user are provided below under 'Relevant Past Conversation Context:'.
Please carefully consider this past context to inform your current answer and maintain conversational coherence.
If the past context is not directly relevant or insufficient for the current question, answer the user's current question to the best of your general knowledge.
Your primary goal is to be helpful, accurate, and maintain a natural conversational flow.
"""
DEFAULT_TEMPERATURE_QWEN3 = 0.6 
DEFAULT_MAX_TOKENS_QWEN3 = 2048
DEFAULT_NUM_RELEVANT_PAST_MESSAGES = 2 
DEFAULT_RECENT_HISTORY_FOR_FLOW_COUNT = 6

QWEN3_SIZE_GB_APPROX = 9.1 
NOMIC_EMBED_SIZE_GB_APPROX = 0.3 
MAX_VRAM_GB = 16
MAX_RESPONSE_DISPLAY_LENGTH = 300

# --- API Communication & Utility Functions ---

@st.cache_data(ttl=3600) 
def get_available_lm_studio_models_with_details(api_base_url: str, api_key: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Fetches the list of available models (with their full details) from LM Studio's /v1/models endpoint.
    Returns a tuple: (list of model objects, error_message_string or None)
    """
    if not api_base_url:
        return [], "API Base URL is not set. Please configure it in the sidebar."
    
    try:
        parsed_base = urlparse(api_base_url)
        if not (parsed_base.scheme and parsed_base.netloc):
            raise ValueError("Invalid API Base URL format. Must include scheme (e.g., http) and host.")
        models_endpoint = urljoin(api_base_url, "models") 
    except Exception as e:
        return [], f"Invalid API Base URL ('{api_base_url}'). Error constructing /models endpoint: {str(e)}"

    headers: Dict[str, str] = {}
    if api_key: headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.get(models_endpoint, headers=headers, timeout=10) 
        response.raise_for_status()
        models_data: Any = response.json()
        
        model_objects: List[Dict[str, Any]] = []
        candidate_list: List[Dict[str, Any]] = []

        if isinstance(models_data, dict):
            candidate_list = models_data.get("data", models_data.get("models", [])) 
        elif isinstance(models_data, list):
            candidate_list = models_data
        
        for entry in candidate_list:
            if isinstance(entry, dict) and entry.get("id") and isinstance(entry.get("id"), str) and entry.get("id").strip():
                model_objects.append(entry) # Store the whole model object
        
        if response.status_code == 200 and not model_objects and not candidate_list:
            return [], None # Success, but no models available/found
        if not model_objects and response.status_code == 200 :
             return [], f"Models endpoint responded successfully but no valid model entries found. Raw: {str(models_data)[:MAX_RESPONSE_DISPLAY_LENGTH]}"
        if not model_objects: 
            return [], f"Failed to extract model entries from {models_endpoint}. Unexpected response. Raw: {str(models_data)[:MAX_RESPONSE_DISPLAY_LENGTH]}"
        return model_objects, None
    except requests.exceptions.Timeout:
        return [], f"Timeout fetching models from {models_endpoint}. LM Studio server unresponsive."
    except requests.exceptions.ConnectionError:
        return [], f"Connection error to {models_endpoint}. Is LM Studio server running and accessible via this URL?"
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401: return [], f"HTTP Error 401: Unauthorized for {models_endpoint}. Check API Key."
        if e.response.status_code == 404: return [], f"HTTP Error 404: Models endpoint not found at {models_endpoint}. Verify your API Base URL."
        if e.response.status_code == 429: return [], f"HTTP Error 429: Too Many Requests to {models_endpoint}."
        return [], f"HTTP Error {e.response.status_code} from {models_endpoint}: {e.response.text[:MAX_RESPONSE_DISPLAY_LENGTH]}"
    except json.JSONDecodeError:
        raw_text = response.text if 'response' in locals() else "N/A"
        return [], f"Error parsing JSON from {models_endpoint}. Raw: {raw_text[:MAX_RESPONSE_DISPLAY_LENGTH]}"
    except requests.exceptions.RequestException as e:
        return [], f"Request error fetching models from {models_endpoint}: {str(e)}"
    except Exception as e:
        return [], f"Unhandled error fetching models: {str(e)}"

@st.cache_data(max_entries=512) 
def get_nomic_embedding(api_base_url: str, text_to_embed: str, api_key: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Gets text embeddings. Returns (normalized_numpy_array, None) or (None, error_message)."""
    if not api_base_url: return None, "Error: API Base URL not configured."
    if not text_to_embed.strip(): return None, "Error: Text to embed cannot be empty."

    embedding_endpoint = urljoin(api_base_url, "embeddings")
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key: headers["Authorization"] = f"Bearer {api_key}"
    
    payload: Dict[str, Any] = {"model": NOMIC_EMBEDDING_MODEL_ID, "input": [text_to_embed]}
    try:
        response = requests.post(embedding_endpoint, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        embedding_data: Any = response.json()
        if "data" in embedding_data and isinstance(embedding_data["data"], list) and \
           len(embedding_data["data"]) > 0 and "embedding" in embedding_data["data"][0] and \
           isinstance(embedding_data["data"][0]["embedding"], list) :
            vector = np.array(embedding_data["data"][0]["embedding"], dtype=np.float32)
            # L2 Normalization (Your Suggestion 2 for Embedding Normalization)
            norm = np.linalg.norm(vector)
            if norm == 0: return vector, None # Avoid division by zero, return original (zero vector)
            return vector / norm, None
        return None, f"Unexpected response structure for embeddings. Raw: {str(embedding_data)[:MAX_RESPONSE_DISPLAY_LENGTH]}"
    except requests.exceptions.RequestException as e: return None, f"Error getting Nomic embedding: {str(e)}"
    except Exception as e: return None, f"Unexpected error in get_nomic_embedding: {str(e)}"

def get_qwen3_rag_chat_response_stream(api_base_url: str, messages_payload_with_context: List[Dict[str,str]], 
                                     temperature: float, max_tokens: Optional[int], 
                                     api_key: Optional[str] = None) -> Iterator[str]:
    """Yields Qwen3 chat response content using streaming."""
    if not api_base_url: yield "Error: API Base URL not configured."; return

    chat_endpoint = urljoin(api_base_url, "chat/completions")
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key: headers["Authorization"] = f"Bearer {api_key}"
    
    payload: Dict[str, Any] = {
        "model": QWEN3_CHAT_MODEL_ID, "messages": messages_payload_with_context,
        "temperature": temperature, "max_tokens": max_tokens, "stream": True
    }
    try:
        with requests.post(chat_endpoint, headers=headers, json=payload, stream=True, timeout=(10, 300)) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        json_data_str = decoded_line[len('data: '):].strip()
                        if json_data_str == "[DONE]": break
                        if json_data_str:
                            try:
                                chunk = json.loads(json_data_str)
                                content_piece = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                                if content_piece: yield content_piece
                            except json.JSONDecodeError: pass 
    except requests.exceptions.RequestException as e: yield f"Error during Qwen3 RAG chat: {str(e)}"
    except Exception as e: yield f"Unexpected error during Qwen3 RAG chat: {str(e)}"

def find_relevant_past_messages(query_embedding: Optional[np.ndarray], 
                                conversation_history: List[Dict[str, Any]], 
                                top_n: int) -> List[str]:
    """Finds top_n most similar past messages. Embeddings should already be normalized."""
    if query_embedding is None or not conversation_history or top_n == 0: return []

    embeddable_history = [msg for msg in conversation_history if msg.get("embedding") is not None and isinstance(msg.get("embedding"), np.ndarray)]
    if not embeddable_history: return []

    history_embeddings = np.array([msg["embedding"] for msg in embeddable_history])
    
    if history_embeddings.ndim == 1: 
        if history_embeddings.shape[0] == query_embedding.shape[0]:
             history_embeddings = history_embeddings.reshape(1, -1)
        else: return []

    query_embedding_reshaped = query_embedding.reshape(1, -1) if query_embedding.ndim == 1 else query_embedding
    if history_embeddings.shape[0] == 0: return []
    if history_embeddings.shape[1] != query_embedding_reshaped.shape[1]:
        st.sidebar.warning(f"Embedding dimension mismatch: Query ({query_embedding_reshaped.shape[1]}) vs History ({history_embeddings.shape[1]}). Skipping RAG.", icon="⚠️")
        return []
        
    try:
        similarities = cosine_similarity(query_embedding_reshaped, history_embeddings)[0]
        actual_top_n = min(top_n, len(similarities))
        if actual_top_n == 0: return []
        relevant_indices = np.argsort(similarities)[-actual_top_n:][::-1] 
        return [f"{embeddable_history[idx]['role'].capitalize()}: {embeddable_history[idx]['content']}" for idx in relevant_indices]
    except Exception as e:
        st.sidebar.warning(f"Similarity calculation error: {e}. Skipping RAG.", icon="⚠️")
        return []

# --- Streamlit UI ---
st.set_page_config(page_title="Contextual Qwen3 Chat", layout="wide", initial_sidebar_state="expanded")
st.title("🧠 Superpowered Qwen3-14B Chat")
st.caption(f"Utilizing {QWEN3_CHAT_MODEL_ID} (Chat) and {NOMIC_EMBEDDING_MODEL_ID} (Embeddings for Context) via LM Studio.")

# --- Session State Initialization ---
default_ss = {
    "chat_messages": [], "api_base_url": LM_STUDIO_DEFAULT_BASE_URL, "api_key": "", 
    "qwen3_temperature": DEFAULT_TEMPERATURE_QWEN3, "qwen3_max_tokens": DEFAULT_MAX_TOKENS_QWEN3,
    "system_prompt": DEFAULT_SYSTEM_PROMPT_RAG,
    "num_relevant_past_messages": DEFAULT_NUM_RELEVANT_PAST_MESSAGES,
    "recent_history_for_flow_count": DEFAULT_RECENT_HISTORY_FOR_FLOW_COUNT,
    "all_available_models_details": [], # Will store list of model dicts
    "models_error_msg": None, "attempted_model_check": False, 
    "qwen3_details": None, "nomic_details": None, # For storing full details of required models
    "vram_check_displayed": False, "last_known_api_base_url": "", "last_known_api_key": "", 
    "show_rag_context_debug": False,
}
for k, v in default_ss.items():
    if k not in st.session_state: st.session_state[k] = v

# --- Helper Function to Load and Check Required Models ---
def _check_and_set_models_availability(force_refresh: bool = False):
    """Checks for required models in LM Studio and updates session state."""
    if force_refresh: get_available_lm_studio_models_with_details.clear() 

    if not st.session_state.api_base_url:
        st.session_state.models_error_msg = "API Base URL is not configured."
        st.session_state.qwen3_details = None; st.session_state.nomic_details = None
        st.session_state.attempted_model_check = True
        return

    with st.sidebar.spinner("Checking LM Studio model availability..."):
        model_objects, error = get_available_lm_studio_models_with_details(st.session_state.api_base_url, st.session_state.api_key)
        st.session_state.attempted_model_check = True
        if error:
            st.session_state.models_error_msg = error 
            st.session_state.all_available_models_details = []
            st.session_state.qwen3_details = None; st.session_state.nomic_details = None
        else:
            st.session_state.models_error_msg = None 
            st.session_state.all_available_models_details = model_objects
            # Find specific details for our required models (Your Suggestion 1 for Model Metadata)
            st.session_state.qwen3_details = next((m for m in model_objects if m.get("id") == QWEN3_CHAT_MODEL_ID), None)
            st.session_state.nomic_details = next((m for m in model_objects if m.get("id") == NOMIC_EMBEDDING_MODEL_ID), None)
    
    st.session_state.last_known_api_base_url = st.session_state.api_base_url
    st.session_state.last_known_api_key = st.session_state.api_key

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    new_api_base_url = st.text_input("LM Studio API Base URL:", value=st.session_state.api_base_url, help=f"E.g., '{LM_STUDIO_DEFAULT_BASE_URL}'.")
    new_api_key = st.text_input("API Key (Optional):", value=st.session_state.api_key, type="password")

    if new_api_base_url != st.session_state.last_known_api_base_url or new_api_key != st.session_state.last_known_api_key:
        st.session_state.api_base_url = new_api_base_url
        st.session_state.api_key = new_api_key
        st.session_state.attempted_model_check = False
        get_available_lm_studio_models_with_details.clear()

    if not st.session_state.attempted_model_check and st.session_state.api_base_url:
        _check_and_set_models_availability()
        st.rerun()

    if st.button("🔄 Check/Refresh Model Availability", help="Verifies connection and required models."):
        _check_and_set_models_availability(force_refresh=True)
        st.rerun()

    if st.session_state.attempted_model_check:
        if st.session_state.models_error_msg: st.error(f"{st.session_state.models_error_msg}")
        else:
            st.success("LM Studio connection check complete.")
            # Display details for Qwen3 (Your Suggestion 1 for Model Metadata)
            if st.session_state.qwen3_details:
                with st.expander(f"✅ {QWEN3_CHAT_MODEL_ID} (Chat) Details", expanded=False):
                    st.json(st.session_state.qwen3_details)
            else: st.warning(f"⚠️ {QWEN3_CHAT_MODEL_ID} (Chat) NOT FOUND.")
            # Display details for Nomic (Your Suggestion 1 for Model Metadata)
            if st.session_state.nomic_details:
                with st.expander(f"✅ {NOMIC_EMBEDDING_MODEL_ID} (Embed) Details", expanded=False):
                    st.json(st.session_state.nomic_details)
            else: st.warning(f"⚠️ {NOMIC_EMBEDDING_MODEL_ID} (Embed) NOT FOUND.")
            if not st.session_state.all_available_models_details and not st.session_state.models_error_msg :
                 st.info("No models were listed by the API.")
    elif not st.session_state.api_base_url:
        st.warning("Please enter API Base URL to check models.")

    st.markdown("---")
    st.header(f"💬 {QWEN3_CHAT_MODEL_ID} Settings")
    st.session_state.system_prompt = st.text_area("System Prompt (RAG):", value=st.session_state.system_prompt, height=200)
    st.session_state.qwen3_temperature = st.slider("Temperature:", 0.0, 2.0, st.session_state.qwen3_temperature, 0.05)
    st.session_state.qwen3_max_tokens = st.number_input("Max Response Tokens:", 0, 32768, st.session_state.qwen3_max_tokens, 64)
    
    st.markdown("---")
    st.header("🧠 RAG Context Settings")
    st.session_state.num_relevant_past_messages = st.slider(
        "Relevant Past Messages to Retrieve:", 0, 10, st.session_state.num_relevant_past_messages, 1)
    st.session_state.recent_history_for_flow_count = st.slider(
        "Recent Messages for Conversational Flow:", 0, 20, st.session_state.recent_history_for_flow_count, 2)
    st.session_state.show_rag_context_debug = st.checkbox("Show Retrieved RAG Context (Debug)", value=st.session_state.show_rag_context_debug)

    st.markdown("---")
    if st.button("🗑️ Clear Chat History & Reset Prompt", help="Clears conversation and resets system prompt."):
        st.session_state.chat_messages = []
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT_RAG 
        st.success("Chat history and system prompt reset!")
        st.rerun()

# VRAM Fit Check
if not st.session_state.vram_check_displayed:
    if QWEN3_SIZE_GB_APPROX > MAX_VRAM_GB: st.toast(f"Warning: {QWEN3_CHAT_MODEL_ID} ({QWEN3_SIZE_GB_APPROX}GB est.) might challenge VRAM.", icon="⚠️")
    if NOMIC_EMBED_SIZE_GB_APPROX > MAX_VRAM_GB: st.toast(f"Warning: {NOMIC_EMBEDDING_MODEL_ID} ({NOMIC_EMBED_SIZE_GB_APPROX}GB est.) might challenge VRAM.", icon="⚠️")
    st.session_state.vram_check_displayed = True

# --- Main Chat Application Area ---
st.header(f"💬 Chat with {QWEN3_CHAT_MODEL_ID} (Dynamic Contextual Memory)")

for msg_data in st.session_state.chat_messages:
    with st.chat_message(msg_data["role"]): st.markdown(msg_data["content"])

if user_prompt := st.chat_input(f"Ask {QWEN3_CHAT_MODEL_ID}..."):
    if not st.session_state.api_base_url: st.error("⚠️ API Base URL is not configured.")
    elif not st.session_state.qwen3_details: st.error(f"⚠️ Chat model {QWEN3_CHAT_MODEL_ID} not available. Check LM Studio.")
    elif not st.session_state.nomic_details: st.error(f"⚠️ Embedding model {NOMIC_EMBEDDING_MODEL_ID} not available. Required for RAG. Check LM Studio.")
    else:
        user_prompt_embedding: Optional[np.ndarray] = None
        with st.spinner("Embedding your message..."):
            embedding_vector, embed_err = get_nomic_embedding(st.session_state.api_base_url, user_prompt, st.session_state.api_key)
        if embed_err: st.warning(f"Could not embed message: {embed_err}. Proceeding without RAG context.")
        else: user_prompt_embedding = embedding_vector
        
        st.session_state.chat_messages.append({"role": "user", "content": user_prompt, "embedding": user_prompt_embedding})
        with st.chat_message("user"): st.markdown(user_prompt)

        relevant_context_texts: List[str] = []
        if user_prompt_embedding is not None and st.session_state.num_relevant_past_messages > 0:
            with st.spinner("Finding relevant past conversation..."):
                relevant_context_texts = find_relevant_past_messages(user_prompt_embedding, st.session_state.chat_messages[:-1], st.session_state.num_relevant_past_messages)
        
        current_system_prompt: str = st.session_state.system_prompt
        context_block_for_display: str = ""
        if relevant_context_texts:
            # Your Suggestion 3 for RAG Context Count Display
            context_block_for_display = f"\n\n--- Retrieved {len(relevant_context_texts)} Relevant Past Message(s) for Context ---\n" + "\n".join(f"- {text}" for text in reversed(relevant_context_texts))
            current_system_prompt += context_block_for_display # Augment the system prompt for the LLM
        
        if st.session_state.show_rag_context_debug:
            with st.sidebar.expander("Retrieved RAG Context (for current turn)", expanded=True): # Default to expanded
                if context_block_for_display: st.markdown(context_block_for_display)
                else: st.caption("No relevant context found or RAG retrieval count is zero.")
        
        text_chat_history: List[Dict[str,str]] = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.chat_messages]
        recent_raw_text_history_count: int = st.session_state.recent_history_for_flow_count 
        recent_history_for_payload = text_chat_history[-(recent_raw_text_history_count + 1):] if recent_raw_text_history_count > 0 else ([text_chat_history[-1]] if text_chat_history else [])

        qwen3_payload: List[Dict[str,str]] = [{"role": "system", "content": current_system_prompt}] + recent_history_for_payload
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_qwen3_response: str = ""
            error_in_stream: bool = False
            spinner_msg = f"Asking {QWEN3_CHAT_MODEL_ID}..."
            with st.spinner(spinner_msg):
                try:
                    for chunk in get_qwen3_rag_chat_response_stream(
                        st.session_state.api_base_url, qwen3_payload,
                        st.session_state.qwen3_temperature, st.session_state.qwen3_max_tokens,
                        st.session_state.api_key):
                        if chunk.startswith("Error:"): full_qwen3_response = chunk; error_in_stream = True; break 
                        full_qwen3_response += chunk
                        message_placeholder.markdown(full_qwen3_response + "▌")
                    message_placeholder.markdown(full_qwen3_response)
                except Exception as e: 
                    full_qwen3_response = f"Fatal error during response streaming: {str(e)}"
                    message_placeholder.error(full_qwen3_response); error_in_stream = True
            
            assistant_response_embedding: Optional[np.ndarray] = None
            if not error_in_stream and full_qwen3_response:
                with st.spinner("Embedding assistant's response..."):
                    embedding_vector, embed_err = get_nomic_embedding(st.session_state.api_base_url, full_qwen3_response, st.session_state.api_key)
                if embed_err: st.warning(f"Could not embed assistant's response: {embed_err}.")
                else: assistant_response_embedding = embedding_vector
                st.session_state.chat_messages.append({"role": "assistant", "content": full_qwen3_response, "embedding": assistant_response_embedding})
            elif error_in_stream: 
                # Error already displayed by message_placeholder. User's message remains.
                pass
            elif not full_qwen3_response : 
                message_placeholder.warning(f"{QWEN3_CHAT_MODEL_ID} returned an empty response.")
        st.rerun()
