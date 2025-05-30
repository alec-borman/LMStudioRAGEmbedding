import streamlit as st
import requests
import json
from urllib.parse import urlparse, urljoin
from typing import List, Tuple, Optional, Dict, Any, Iterator, overload # Added overload
import logging # Step 7: Structured Logging
import os # For API Key from environment
from dotenv import load_dotenv # For local API Key management

# --- Application Configuration (Could be in a config.py module) ---
# Report Section 4: LM Studio API Configuration. Using API Base URL.
LM_STUDIO_DEFAULT_BASE_URL = "http://localhost:1234/v1" 
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."
MAX_RESPONSE_DISPLAY_LENGTH = 300 

# Step 7: Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file for local development
load_dotenv()

# --- API Communication Logic (Could be in an api_client.py module) ---

# Report Section 8 & Step 6: Caching
@st.cache_data(ttl=3600) # Cache model list for 1 hour
def get_available_models(api_base_url: str, api_key: Optional[str] = None) -> Tuple[List[str], Optional[str]]:
    """
    Fetches the list of available model IDs from the LM Studio API's /models endpoint.
    This function focuses on data retrieval and parsing, returning data or error info.
    UI interactions are handled by the calling UI code.
    (Addresses Report Sections 3, 4.1, Major Improvements, Critical Fixes for return syntax)

    Args:
        api_base_url (str): The base URL of the LM Studio API (e.g., "http://localhost:1234/v1").
        api_key (Optional[str]): An optional API key for authentication.

    Returns:
        Tuple[List[str], Optional[str]]: (list_of_model_ids, error_message_string_or_None)
    """
    if not api_base_url:
        return [], "API Base URL is not set. Please provide a valid LM Studio API Base URL in the sidebar."

    try:
        parsed_base = urlparse(api_base_url)
        if not (parsed_base.scheme and parsed_base.netloc): # Basic validation
            raise ValueError("Invalid API Base URL format. Must include scheme (e.g., http or https) and host.")
        # Ensure base URL ends with a slash for urljoin to work reliably with relative paths
        base_url_with_slash = api_base_url if api_base_url.endswith('/') else api_base_url + '/'
        models_endpoint = urljoin(base_url_with_slash, "models")
    except Exception as e:
        logging.error(f"Error constructing models endpoint from base URL '{api_base_url}': {e}")
        return [], f"Invalid API Base URL ('{api_base_url}'). Error: {str(e)}"

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key: headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.get(models_endpoint, headers=headers, timeout=10) 
        response.raise_for_status() 
        models_data: Any = response.json()
        model_ids: List[str] = []
        
        model_list_candidates: List[Dict[str, Any]] = []
        if isinstance(models_data, dict):
            candidate_list_from_data = models_data.get("data", [])
            if isinstance(candidate_list_from_data, list): candidate_list.extend(candidate_list_from_data)
            if not model_list_candidates and "models" in models_data and isinstance(models_data["models"], list):
                 candidate_list.extend(models_data["models"])
        elif isinstance(models_data, list): 
            model_list_candidates = models_data
        
        for model_entry in model_list_candidates:
            if isinstance(model_entry, dict): 
                model_id = model_entry.get("id")
                if model_id and isinstance(model_id, str) and model_id.strip():
                    model_ids.append(model_id)
        
        # Critical Fix (Report Item 1, Syntax Error in get_available_models return)
        if response.status_code == 200 and not model_ids and not model_list_candidates:
            return [], None 
        if not model_ids and response.status_code == 200 :
             return [], f"Models endpoint responded successfully but no valid model IDs were extracted. Raw: {str(models_data)[:MAX_RESPONSE_DISPLAY_LENGTH]}"
        if not model_ids: 
            return [], f"Failed to extract model IDs from {models_endpoint}. Unexpected response. Raw: {str(models_data)[:MAX_RESPONSE_DISPLAY_LENGTH]}"
        return model_ids, None
    except requests.exceptions.Timeout:
        logging.warning(f"Timeout fetching models from {models_endpoint}.")
        return [], f"Timeout when fetching models from {models_endpoint}. Server unresponsive."
    except requests.exceptions.ConnectionError:
        logging.error(f"Connection error for {models_endpoint}.")
        return [], f"Connection error for {models_endpoint}. Is LM Studio server running?"
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        err_text = e.response.text[:MAX_RESPONSE_DISPLAY_LENGTH]
        logging.error(f"HTTP Error {status_code} from {models_endpoint}: {err_text}", exc_info=True)
        if status_code == 401: return [], f"HTTP 401: Unauthorized for {models_endpoint}. Check API Key."
        if status_code == 404: return [], f"HTTP 404: Models endpoint not found at {models_endpoint}. Verify API Base URL."
        if status_code == 429: return [], f"HTTP 429: Too Many Requests to {models_endpoint}."
        return [], f"HTTP Error {status_code} from {models_endpoint}: {err_text}"
    except json.JSONDecodeError as e_json:
        raw_text = response.text if 'response' in locals() else "N/A"
        logging.error(f"JSONDecodeError from {models_endpoint}. Raw: {raw_text[:MAX_RESPONSE_DISPLAY_LENGTH]}", exc_info=True)
        return [], f"Error parsing JSON from {models_endpoint}. Response: {raw_text[:MAX_RESPONSE_DISPLAY_LENGTH]}"
    except requests.exceptions.RequestException as e_req:
        logging.error(f"RequestException for {models_endpoint}: {e_req}", exc_info=True)
        return [], f"Network/Request error for {models_endpoint}: {str(e_req)}"
    except Exception as e_unhandled:
        logging.error(f"Unhandled error fetching models from {models_endpoint}: {e_unhandled}", exc_info=True)
        return [], f"Unhandled error fetching models: {str(e_unhandled)}"

# Step 8 & User Request: Type Hinting for generate_chat_response with @overload
@overload
def generate_chat_response(
    api_base_url: str, model_id: str, messages_payload: List[Dict[str, str]], 
    temperature: float, max_tokens: Optional[int], stream: लिटरल[True], # type: ignore
    api_key: Optional[str] = None
) -> Iterator[str]: ...

@overload
def generate_chat_response(
    api_base_url: str, model_id: str, messages_payload: List[Dict[str, str]], 
    temperature: float, max_tokens: Optional[int], stream: लिटरल[False], # type: ignore
    api_key: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]: ...

def generate_chat_response(
    api_base_url: str, 
    model_id: str, 
    messages_payload: List[Dict[str, str]], 
    temperature: float, 
    max_tokens: Optional[int], 
    stream: bool, 
    api_key: Optional[str] = None
) -> Any: # Actual implementation uses Any for simplicity with overload
    """
    Generates chat response from LM Studio API.
    """
    if not model_id:
        error_msg = "Error: No model selected. Please select a model in the sidebar."
        return iter([error_msg]) if stream else (None, error_msg)
    if not api_base_url:
        error_msg = "Error: API Base URL not configured in the sidebar."
        return iter([error_msg]) if stream else (None, error_msg)

    base_url_with_slash = api_base_url if api_base_url.endswith('/') else api_base_url + '/'
    chat_completions_endpoint = urljoin(base_url_with_slash, "chat/completions")
    
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key: headers["Authorization"] = f"Bearer {api_key}"

    payload: Dict[str, Any] = {
        "model": model_id, "messages": messages_payload, "temperature": temperature,
        "max_tokens": max_tokens if max_tokens and max_tokens > 0 else None, 
        "stream": stream
    }

    if stream:
        def _stream_generator() -> Iterator[str]:
            try:
                with requests.post(chat_completions_endpoint, headers=headers, json=payload, stream=True, timeout=(10, 300)) as response_iter:
                    response_iter.raise_for_status()
                    for line in response_iter.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            if decoded_line.startswith('data: '):
                                json_data_str = decoded_line[len('data: '):].strip()
                                if json_data_str == "[DONE]": break
                                if json_data_str:
                                    try:
                                        chunk = json.loads(json_data_str)
                                        choice = chunk.get("choices", [{}])[0] 
                                        delta = choice.get("delta", {})
                                        content_piece = delta.get("content")
                                        if content_piece: yield content_piece
                                    except json.JSONDecodeError as json_e:
                                        # Step 7: Log JSONDecodeError in streaming
                                        logging.warning(f"Malformed JSON chunk in stream: '{json_data_str}'. Error: {json_e}")
                                        # Optionally yield an error marker if UI needs to know
                                        # yield "[Stream Error: Malformed data chunk]"
                                        pass 
                                    except (IndexError, KeyError) as e_struct:
                                        logging.warning(f"Unexpected SSE chunk structure in stream: {chunk}. Error: {e_struct}")
                                        pass 
            except requests.exceptions.Timeout: 
                logging.warning(f"Stream timeout for {model_id} at {chat_completions_endpoint}.")
                yield f"Error (Stream): Timeout for {model_id}."
            except requests.exceptions.ConnectionError: 
                logging.error(f"Stream connection error for {model_id} at {chat_completions_endpoint}.")
                yield f"Error (Stream): Connection failed for {model_id}."
            except requests.exceptions.HTTPError as e_http:
                err_detail = e_http.response.text[:MAX_RESPONSE_DISPLAY_LENGTH] if e_http.response else "N/A"
                logging.error(f"Stream HTTP Error {e_http.response.status_code} for {model_id}: {err_detail}", exc_info=True)
                if e_http.response.status_code == 401: yield f"Error (Stream): HTTP 401 Unauthorized. Check API Key."; return
                if e_http.response.status_code == 429: yield f"Error (Stream): HTTP 429 Too Many Requests."; return
                yield f"Error (Stream): HTTP {e_http.response.status_code}: {err_detail}"
            except requests.exceptions.RequestException as e_req: 
                logging.error(f"Stream RequestException for {model_id}: {e_req}", exc_info=True)
                yield f"Error (Stream): Network issue for {model_id}: {str(e_req)}"
            except Exception as e_gen: 
                logging.error(f"Stream Unhandled issue for {model_id}: {e_gen}", exc_info=True)
                yield f"Error (Stream): Unhandled issue for {model_id}: {str(e_gen)}"
        return _stream_generator()
    else: # Non-streaming
        try:
            response = requests.post(chat_completions_endpoint, headers=headers, json=payload, timeout=180) 
            response.raise_for_status()
            assistant_response = response.json()
            content = assistant_response.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content, None
        except requests.exceptions.Timeout: 
            logging.warning(f"Non-stream timeout for {model_id} at {chat_completions_endpoint}.")
            return None, f"Error: Timeout for {model_id} (non-streaming)."
        except requests.exceptions.ConnectionError: 
            logging.error(f"Non-stream connection error for {model_id} at {chat_completions_endpoint}.")
            return None, f"Error: Connection failed for {model_id} (non-streaming)."
        except requests.exceptions.HTTPError as e_http:
            err_detail = e_http.response.text[:MAX_RESPONSE_DISPLAY_LENGTH] if e_http.response else "N/A"
            logging.error(f"Non-stream HTTP Error {e_http.response.status_code} for {model_id}: {err_detail}", exc_info=True)
            if e_http.response.status_code == 401: return None, f"Error: HTTP 401 Unauthorized. Check API Key."
            if e_http.response.status_code == 429: return None, f"Error: HTTP 429 Too Many Requests."
            return None, f"Error: HTTP {e_http.response.status_code}: {err_detail}"
        except (json.JSONDecodeError, KeyError, IndexError) as e_parse:
            raw_text = response.text if 'response' in locals() else 'N/A'
            logging.error(f"Non-stream parsing error for {model_id}: {e_parse}. Raw: {raw_text[:1000]}", exc_info=True) # Log more raw text
            return None, f"Error parsing response for {model_id}: {str(e_parse)}. Raw: {raw_text[:MAX_RESPONSE_DISPLAY_LENGTH]}"
        except requests.exceptions.RequestException as e_req: 
            logging.error(f"Non-stream RequestException for {model_id}: {e_req}", exc_info=True)
            return None, f"Error: Network issue for {model_id}: {str(e_req)}"
        except Exception as e_gen: 
            logging.error(f"Non-stream Unhandled issue for {model_id}: {e_gen}", exc_info=True)
            return None, f"Error: Unhandled issue for {model_id}: {str(e_gen)}"

# --- Streamlit UI ---
st.set_page_config(page_title="LMChat Studio Interface", layout="wide", initial_sidebar_state="expanded")
st.title("LMChat Studio Interface 🤖")
st.caption("Interact with your locally hosted Large Language Models via LM Studio.")

# --- Session State Initialization (Critical Fix 1) ---
default_ss_values: Dict[str, Any] = {
    "messages": [], "api_base_url": LM_STUDIO_DEFAULT_BASE_URL, "api_key": "", 
    "system_prompt": DEFAULT_SYSTEM_PROMPT, "temperature": 0.7, "max_tokens": 512,      
    "selected_model": None, "available_models": [], 
    "models_error": None, "models_loaded_for_current_config": False,
    "enable_streaming": True, "last_known_api_base_url": "", "last_known_api_key": "" 
}
for key, value in default_ss_values.items():
    if key not in st.session_state: st.session_state[key] = value

# --- Helper Function to Load Models and Update UI ---
def _load_and_update_models_sidebar(force_refresh: bool = False):
    """Loads models, updates session state, and provides UI feedback in the sidebar."""
    if force_refresh: get_available_models.clear()

    if not st.session_state.api_base_url:
        st.session_state.models_error = "API Base URL is not configured. Please enter it."
        st.session_state.available_models = []; st.session_state.selected_model = None
        st.session_state.models_loaded_for_current_config = False
        return

    with st.sidebar.spinner("Fetching available models..."):
        # Use the API key from session state for get_available_models
        models, error = get_available_models(st.session_state.api_base_url, st.session_state.api_key)
        st.session_state.models_loaded_for_current_config = True
        
        if error:
            st.session_state.models_error = error 
            st.session_state.available_models = []; st.session_state.selected_model = None 
        else: 
            st.session_state.models_error = None 
            st.session_state.available_models = models 
            if models: 
                current_sel = st.session_state.selected_model
                if current_sel not in models or current_sel is None:
                    st.session_state.selected_model = models[0] 
                st.sidebar.success("Models loaded!" if not force_refresh else "Models refreshed!")
            else: 
                st.session_state.selected_model = None 
                st.sidebar.info("LM Studio API OK, but no models were listed.")
    
    st.session_state.last_known_api_base_url = st.session_state.api_base_url
    st.session_state.last_known_api_key = st.session_state.api_key

# --- Sidebar for Settings ---
with st.sidebar:
    st.header("⚙️ Connection & Model")
    
    # API Configuration
    # User Request: API Key Security - Prioritize environment variable, then secrets, then input as last resort.
    env_api_key = os.getenv("LM_STUDIO_API_KEY")
    secrets_api_key = st.secrets.get("LM_STUDIO_API_KEY") if hasattr(st, "secrets") else None
    
    # Determine initial API key value for the input field
    initial_api_key_for_input = st.session_state.api_key # Use session state if already set by user
    if not initial_api_key_for_input and env_api_key:
        initial_api_key_for_input = env_api_key
    elif not initial_api_key_for_input and secrets_api_key:
        initial_api_key_for_input = secrets_api_key

    new_api_base_url = st.text_input("LM Studio API Base URL:", value=st.session_state.api_base_url, 
                                   help=f"E.g., '{LM_STUDIO_DEFAULT_BASE_URL}'. Must end with '/v1' or be the server root.")
    new_api_key_input = st.text_input("API Key (Manual Override):", value=initial_api_key_for_input, type="password",
                                 help="Leave blank to use environment/secrets key. Overrides if set.")

    # Determine effective API key: User input > Secrets > Environment > Session State (if not from input)
    effective_api_key = new_api_key_input # User input takes precedence if provided
    if not effective_api_key and secrets_api_key: # Check secrets if input is blank
        effective_api_key = secrets_api_key
    if not effective_api_key and env_api_key: # Check environment if secrets/input blank
        effective_api_key = env_api_key
    
    # Update session state with the determined effective API key for internal use
    # and the URL from input.
    # This ensures that if user clears input, it might fall back to env/secrets.
    # The last_known_api_key tracks what was *used* for the last load attempt.
    
    config_changed = (new_api_base_url != st.session_state.last_known_api_base_url or 
                      effective_api_key != st.session_state.last_known_api_key) # Compare with effective key

    if config_changed:
        st.session_state.api_base_url = new_api_base_url
        st.session_state.api_key = effective_api_key # Store the effective key for API calls
        st.session_state.models_loaded_for_current_config = False 
        get_available_models.clear()

    if st.session_state.api_base_url and not st.session_state.models_loaded_for_current_config:
        _load_and_update_models_sidebar()
        st.rerun() 

    if st.button("🔄 Refresh Models & Test Connection", help="Refreshes model list from LM Studio."):
        # When refreshing, ensure we use the latest values from inputs for the API call
        st.session_state.api_base_url = new_api_base_url 
        st.session_state.api_key = effective_api_key
        _load_and_update_models_sidebar(force_refresh=True)
        st.rerun()

    if st.session_state.models_error: st.error(f"{st.session_state.models_error}")
    elif not st.session_state.api_base_url and st.session_state.models_loaded_for_current_config :
        st.warning("API Base URL was cleared. Please re-enter.")
    elif not st.session_state.api_base_url: 
        st.info("Please enter API Base URL to load models.")
    
    if st.session_state.available_models:
        options = st.session_state.available_models
        current_selection_value = st.session_state.selected_model
        
        default_selectbox_index = 0 
        if current_selection_value in options:
            default_selectbox_index = options.index(current_selection_value)
        elif options: 
            st.session_state.selected_model = options[0] 
            # default_selectbox_index remains 0
        
        if options: # Only show if options exist
            st.session_state.selected_model = st.selectbox(
                "Select Model:", options=options, index=default_selectbox_index,
                key="model_selector_widget_main", 
                help="Choose from models loaded in LM Studio."
            )
            st.caption(f"Using: `{st.session_state.selected_model}`")
    elif st.session_state.api_base_url and st.session_state.models_loaded_for_current_config and not st.session_state.models_error:
        st.info("No models found at the endpoint. Check LM Studio setup.")
    
    if not st.session_state.available_models: st.session_state.selected_model = None

    st.markdown("---")
    st.header("💬 Chat Settings")
    st.session_state.system_prompt = st.text_area("System Prompt:", value=st.session_state.system_prompt, height=100)
    st.session_state.temperature = st.slider("Temperature:", 0.0, 2.0, st.session_state.temperature, 0.05)
    st.session_state.max_tokens = st.number_input("Max Response Tokens (0 for model default):", 0, 32768, st.session_state.max_tokens, 64)
    st.session_state.enable_streaming = st.toggle("Enable Streaming Response", value=st.session_state.enable_streaming)

    st.markdown("---")
    if st.button("✨ New Chat Session", help="Clears chat history and resets system prompt."):
        st.session_state.messages = []
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT 
        st.success("Chat history and system prompt reset!")
        st.rerun()

# --- Main Chat Area ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

if prompt := st.chat_input("Ask your local AI anything..."):
    if not st.session_state.api_base_url: st.error("⚠️ API Base URL not configured.")
    elif not st.session_state.selected_model: st.error("⚠️ No LM Studio model selected.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        api_messages_payload: List[Dict[str, str]] = [{"role": "system", "content": st.session_state.system_prompt}] + \
                               [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response_content: str = ""
            error_occurred: bool = False

            if st.session_state.enable_streaming:
                with st.spinner(f"AI ({st.session_state.selected_model}) is thinking... (streaming)"):
                    try:
                        stream_gen: Iterator[str] = generate_chat_response(
                            st.session_state.api_base_url, st.session_state.selected_model,
                            api_messages_payload, st.session_state.temperature,
                            st.session_state.max_tokens, stream=True, api_key=st.session_state.api_key
                        )
                        for chunk in stream_gen:
                            if chunk.startswith("Error:"): 
                                full_response_content = chunk; error_occurred = True; break 
                            full_response_content += chunk
                            message_placeholder.markdown(full_response_content + "▌")
                        message_placeholder.markdown(full_response_content)
                    except Exception as e: 
                        full_response_content = f"Critical error during streaming setup: {str(e)}"
                        message_placeholder.error(full_response_content); error_occurred = True
            else: 
                with st.spinner(f"AI ({st.session_state.selected_model}) is thinking... (non-streaming)"):
                    # Type cast to Tuple[Optional[str], Optional[str]]
                    response_tuple: Tuple[Optional[str], Optional[str]] = generate_chat_response( # type: ignore
                        st.session_state.api_base_url, st.session_state.selected_model,
                        api_messages_payload, st.session_state.temperature,
                        st.session_state.max_tokens, stream=False, api_key=st.session_state.api_key
                    )
                    ai_content: Optional[str] = response_tuple[0]
                    error_message_str: Optional[str] = response_tuple[1]

                if error_message_str:
                    message_placeholder.error(error_message_str); error_occurred = True
                elif ai_content:
                    message_placeholder.markdown(ai_content); full_response_content = ai_content
                else: 
                    message_placeholder.warning("AI returned an empty response (non-streaming).")
            
            if not error_occurred and full_response_content:
                 st.session_state.messages.append({"role": "assistant", "content": full_response_content})
        st.rerun()
