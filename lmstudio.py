import streamlit as st
import requests
import json
from urllib.parse import urlparse, urljoin
from typing import List, Tuple, Optional, Dict, Any, Iterator

# --- App Configuration (Report Section 4: LM Studio API Configuration) ---
# Using API Base URL for flexibility, specific endpoints will be appended.
LM_STUDIO_DEFAULT_BASE_URL = "http://localhost:1234/v1" 
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."
MAX_RESPONSE_DISPLAY_LENGTH = 300 # For truncating raw error responses in UI

# --- API Communication Logic (Refactored for Robustness & Separation of Concerns) ---

# Report Section 2 & 4: Caching expensive/infrequent operations
@st.cache_data(ttl=3600) # Cache model list for 1 hour
def get_available_models(api_base_url: str, api_key: Optional[str] = None) -> Tuple[List[str], Optional[str]]:
    """
    Fetches the list of available model IDs from the LM Studio API's /models endpoint.
    This function focuses on data retrieval and parsing, returning data or error info.
    UI interactions (like st.warning) should be handled by the calling UI code.
    Returns a tuple: (list_of_model_ids, error_message_string_or_None)
    """
    if not api_base_url:
        return [], "API Base URL is not set. Please provide a valid LM Studio API Base URL."

    try:
        parsed_base = urlparse(api_base_url)
        if not (parsed_base.scheme and parsed_base.netloc):
            raise ValueError("Invalid API Base URL format. Must include scheme (e.g., http) and host.")
        # Report Section 4.1: Correct construction of models_endpoint
        models_endpoint = urljoin(api_base_url + ('/' if not api_base_url.endswith('/') else ''), "models")
    except Exception as e:
        return [], f"Invalid API Base URL ('{api_base_url}'). Error constructing /models endpoint: {str(e)}"

    headers: Dict[str, str] = {}
    if api_key: headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.get(models_endpoint, headers=headers, timeout=10)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        models_data: Any = response.json()
        model_ids: List[str] = []
        
        # Report Section 4.1: Robust API response parsing for OpenAI-compatible /v1/models
        model_list_candidates: List[Dict[str, Any]] = []
        if isinstance(models_data, dict):
            # Prefer "data" key (OpenAI standard), fallback to "models" or other common keys
            candidate_list_from_data = models_data.get("data", [])
            if isinstance(candidate_list_from_data, list):
                model_list_candidates.extend(candidate_list_from_data)
            
            if not model_list_candidates and "models" in models_data and isinstance(models_data["models"], list):
                 model_list_candidates.extend(models_data["models"])
            # Add other potential top-level keys if necessary for LM Studio variants
        elif isinstance(models_data, list): # Direct list of model objects
            model_list_candidates = models_data
        
        for model_entry in model_list_candidates:
            if isinstance(model_entry, dict): # Ensure entry is a dictionary
                model_id = model_entry.get("id")
                if model_id and isinstance(model_id, str) and model_id.strip(): # Ensure ID is a non-empty string
                    model_ids.append(model_id)
        
        # Report Section 4.1: Corrected return for empty but successful responses
        if response.status_code == 200 and not model_ids and not model_list_candidates :
            return [], None # Successfully connected and parsed, but LM Studio reports no models
        
        if not model_ids and response.status_code == 200: # Successfully connected, but structure didn't yield IDs
             return [], f"Models endpoint responded successfully but no valid model IDs were extracted. Raw: {str(models_data)[:MAX_RESPONSE_DISPLAY_LENGTH]}"
        
        if not model_ids: # General failure to extract if not caught above
            # Report Section 4.1: Return empty list and error message (no UI st.warning here)
            return [], f"Failed to extract model IDs from {models_endpoint}. Unexpected response structure. Raw: {str(models_data)[:MAX_RESPONSE_DISPLAY_LENGTH]}"

        return model_ids, None # Successfully fetched model IDs

    # Report Section 4.1 & 6: Corrected syntax for return and specific exception handling
    except requests.exceptions.Timeout:
        return [], f"Timeout when fetching models from {models_endpoint}. LM Studio server might be unresponsive."
    except requests.exceptions.ConnectionError:
        return [], f"Connection error when trying to reach {models_endpoint}. Is LM Studio server running and the URL correct?"
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401: return [], f"HTTP Error 401: Unauthorized for {models_endpoint}. Please check your API Key."
        if e.response.status_code == 404: return [], f"HTTP Error 404: Models endpoint not found at {models_endpoint}. Verify your API Base URL (it should typically end with '/v1')."
        if e.response.status_code == 429: return [], f"HTTP Error 429: Too Many Requests to {models_endpoint}."
        return [], f"HTTP Error {e.response.status_code} from {models_endpoint}: {e.response.text[:MAX_RESPONSE_DISPLAY_LENGTH]}"
    except json.JSONDecodeError:
        raw_text = response.text if 'response' in locals() else "N/A"
        return [], f"Error parsing JSON response from {models_endpoint}. Expected JSON but received: {raw_text[:MAX_RESPONSE_DISPLAY_LENGTH]}"
    except requests.exceptions.RequestException as e:
        return [], f"A network or request error occurred while fetching models from {models_endpoint}: {str(e)}"
    except Exception as e:
        return [], f"An unhandled error occurred while fetching models: {str(e)}"

# Report Section 6: Add Type Hints
def generate_chat_response(
    api_base_url: str, 
    model_id: str, 
    messages_payload: List[Dict[str, str]], 
    temperature: float, 
    max_tokens: Optional[int], 
    stream: bool, 
    api_key: Optional[str] = None
) -> Any: # Returns Iterator[str] for stream, Tuple[Optional[str], Optional[str]] for non-stream
    """
    Generates chat response from LM Studio API.
    If streaming, yields content chunks or error messages (as strings starting with "Error:").
    If not streaming, returns (content_string, error_string_or_None).
    """
    if not model_id:
        error_msg = "Error: No model selected. Please select a model in the sidebar."
        # For stream=True, return a generator that yields the error. For stream=False, return tuple.
        return iter([error_msg]) if stream else (None, error_msg)
    if not api_base_url:
        error_msg = "Error: API Base URL not configured in the sidebar."
        return iter([error_msg]) if stream else (None, error_msg)

    # Report Section 4.1: Correct construction of chat_completions_endpoint
    chat_completions_endpoint = urljoin(api_base_url + ('/' if not api_base_url.endswith('/') else ''), "chat/completions")
    
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": messages_payload,
        "temperature": temperature,
        "max_tokens": max_tokens if max_tokens and max_tokens > 0 else None, 
        "stream": stream
    }

    if stream:
        def _stream_generator() -> Iterator[str]: # Inner generator with type hint
            try:
                # Report Section 4.2: Timeout considerations
                with requests.post(chat_completions_endpoint, headers=headers, json=payload, stream=True, timeout=(10, 300)) as response_iter: # 10s connect, 300s read
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
                                        # Report Section 4.2: Critical fix for streaming JSON parsing
                                        # chunk["choices"] is a LIST. Access its first element.
                                        choice = chunk.get("choices", [{}])[0] 
                                        delta = choice.get("delta", {})
                                        content_piece = delta.get("content")
                                        if content_piece:
                                            yield content_piece
                                    except json.JSONDecodeError:
                                        # Report Section 4.2 & Medium Priority: Log error instead of silent pass
                                        # For a UI app, yielding a warning might be too noisy. Logging is better.
                                        # print(f"DEBUG: Malformed JSON in stream: {json_data_str}") 
                                        pass # Keep pass for smoother UX, but log if needed
                                    except (IndexError, KeyError):
                                        # Handle cases where choices list is empty or delta/content is missing
                                        # print(f"DEBUG: Unexpected chunk structure: {chunk}")
                                        pass
            # Report Section 4.2 & 6: Error handling within the generator
            except requests.exceptions.Timeout:
                yield f"Error: Request to {model_id} at {chat_completions_endpoint} timed out during streaming."
            except requests.exceptions.ConnectionError:
                yield f"Error: Could not connect to {chat_completions_endpoint} for {model_id} during streaming."
            except requests.exceptions.HTTPError as e_http:
                error_detail = e_http.response.text[:MAX_RESPONSE_DISPLAY_LENGTH] if e_http.response else "No response body"
                if e_http.response.status_code == 401: yield f"HTTP Error 401: Unauthorized for {chat_completions_endpoint}. Check API Key." ; return
                if e_http.response.status_code == 429: yield f"HTTP Error 429: Too Many Requests to {chat_completions_endpoint}." ; return
                yield f"HTTP Error {e_http.response.status_code} ({model_id} at {chat_completions_endpoint}): {error_detail}"
            except requests.exceptions.RequestException as e_req:
                yield f"A network or request error occurred for {model_id} at {chat_completions_endpoint} during streaming: {str(e_req)}"
            except Exception as e_gen:
                yield f"An unhandled error occurred during streaming for {model_id} at {chat_completions_endpoint}: {str(e_gen)}"
        return _stream_generator()
    else: # Non-streaming
        try:
            # Report Section 4.3: Timeout for non-streaming
            response = requests.post(chat_completions_endpoint, headers=headers, json=payload, timeout=180) 
            response.raise_for_status()
            assistant_response = response.json()
            # Report Section 4.3: Robust content extraction
            content = assistant_response.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content, None
        except requests.exceptions.Timeout:
            return None, f"Error: Request to {model_id} at {chat_completions_endpoint} timed out (non-streaming)."
        except requests.exceptions.ConnectionError:
            return None, f"Error: Could not connect to {chat_completions_endpoint} for model {model_id} (non-streaming)."
        except requests.exceptions.HTTPError as e_http:
            error_detail = e_http.response.text[:MAX_RESPONSE_DISPLAY_LENGTH] if e_http.response else "No response body"
            if e_http.response.status_code == 401: return None, f"HTTP Error 401: Unauthorized for {chat_completions_endpoint}. Check API Key."
            if e_http.response.status_code == 429: return None, f"HTTP Error 429: Too Many Requests to {chat_completions_endpoint}."
            return None, f"HTTP Error {e_http.response.status_code} ({model_id} at {chat_completions_endpoint}): {error_detail}"
        except (json.JSONDecodeError, KeyError, IndexError) as e_parse:
            raw_text = response.text if 'response' in locals() else 'N/A'
            return None, f"Error parsing non-streaming response from {chat_completions_endpoint} for {model_id}: {str(e_parse)}. Raw: {raw_text[:MAX_RESPONSE_DISPLAY_LENGTH]}"
        except requests.exceptions.RequestException as e_req:
            return None, f"A network or request error occurred for {model_id} at {chat_completions_endpoint} (non-streaming): {str(e_req)}"
        except Exception as e_gen:
            return None, f"An unhandled error occurred (non-streaming) for {model_id} at {chat_completions_endpoint}: {str(e_gen)}"

# --- Streamlit UI ---
st.set_page_config(page_title="LMChat Studio Interface", layout="wide", initial_sidebar_state="expanded")
st.title("LMChat Studio Interface 🤖")
st.caption("Interact with your locally hosted Large Language Models via LM Studio.")

# --- Session State Initialization (Report Section 5.1: Critical Fix) ---
# Ensure all session state variables are initialized correctly if they don't exist
default_ss_values: Dict[str, Any] = {
    "messages": [], 
    "api_base_url": LM_STUDIO_DEFAULT_BASE_URL,
    "api_key": "", 
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "temperature": 0.7,
    "max_tokens": 512,      
    "selected_model": None, 
    "available_models": [], # Correctly initialized as an empty list
    "models_error": None,   
    "models_loaded_for_current_url": False, # Replaces 'models_loaded', more specific
    "enable_streaming": True,
    "last_known_api_base_url": "", 
    "last_known_api_key": "" 
}
for key, value in default_ss_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Helper Function to Load Models and Update UI ---
def _load_and_set_models_ui_feedback(force_refresh: bool = False):
    """
    Loads models from the API, updates session state, and provides UI feedback in the sidebar.
    Report Section 5.2: Optimize model loading trigger
    """
    if force_refresh:
        get_available_models.clear() 

    if not st.session_state.api_base_url:
        st.session_state.models_error = "API Base URL is not configured. Please enter it above."
        st.session_state.available_models = []
        st.session_state.selected_model = None
        st.session_state.models_loaded_for_current_url = False # Mark as not loaded for this (empty) URL
        return

    # Report Section 3: Separation of concerns - UI interaction in UI code
    with st.sidebar.spinner("Fetching available models from LM Studio..."):
        models, error = get_available_models(st.session_state.api_base_url, st.session_state.api_key)
        
        if error:
            st.session_state.models_error = error 
            st.session_state.available_models = [] 
            st.session_state.selected_model = None 
            st.session_state.models_loaded_for_current_url = False
        else: 
            st.session_state.models_error = None 
            st.session_state.available_models = models 
            st.session_state.models_loaded_for_current_url = True # Mark as loaded for current URL
            
            if models: 
                # Report Section 5.2: Corrected model selection logic
                current_selection = st.session_state.selected_model
                if current_selection not in models or current_selection is None:
                    st.session_state.selected_model = models[0] # Default to first model
                # If current_selection is valid and in models, it remains
                st.sidebar.success("Models loaded successfully!" if not force_refresh else "Models refreshed!")
            else: 
                st.session_state.selected_model = None 
                st.sidebar.info("LM Studio connection OK, but no models were found/listed by the API endpoint.")
    
    st.session_state.last_known_api_base_url = st.session_state.api_base_url
    st.session_state.last_known_api_key = st.session_state.api_key

# --- Sidebar for Settings ---
with st.sidebar:
    st.header("⚙️ Connection & Model")
    
    # API Configuration
    current_api_base_url_val = st.session_state.api_base_url
    new_api_base_url_val = st.text_input(
        "LM Studio API Base URL:", 
        value=current_api_base_url_val, 
        help=f"Enter base URL for LM Studio API (e.g., '{LM_STUDIO_DEFAULT_BASE_URL}'). Usually ends with '/v1'."
    )
    current_api_key_val = st.session_state.api_key
    new_api_key_val = st.text_input("API Key (Optional):", value=current_api_key_val, type="password")

    # Detect API URL or Key change to trigger model reload
    # Report Section 5.2: Optimize model loading trigger
    if new_api_base_url_val != st.session_state.last_known_api_base_url or \
       new_api_key_val != st.session_state.last_known_api_key:
        st.session_state.api_base_url = new_api_base_url_val
        st.session_state.api_key = new_api_key_val
        st.session_state.models_loaded_for_current_url = False # Mark as needing reload for new config
        get_available_models.clear() # Clear cache due to config change

    # Load models if URL is set and they haven't been loaded for this URL/key yet
    if st.session_state.api_base_url and not st.session_state.models_loaded_for_current_url:
        _load_and_set_models_ui_feedback()
        st.rerun() # Rerun to update selectbox and status messages

    if st.button("🔄 Refresh Models & Test Connection", help="Refreshes model list from LM Studio."):
        _load_and_set_models_ui_feedback(force_refresh=True)
        st.rerun()

    # Display Model Loading Status/Error
    if st.session_state.models_error:
         st.error(f"{st.session_state.models_error}")
    elif not st.session_state.api_base_url:
        st.warning("Please enter the LM Studio API Base URL to load models.")
    
    # Model Selection Dropdown - only if models are available
    if st.session_state.available_models:
        # Report Section 5.2: Corrected model selection logic and index handling
        # Ensure selected_model is valid within the current available_models list
        chosen_model = st.session_state.selected_model
        if chosen_model not in st.session_state.available_models:
            chosen_model = st.session_state.available_models[0] # Default to first if stale or None
            st.session_state.selected_model = chosen_model # Update session state

        try:
            current_selection_index = st.session_state.available_models.index(chosen_model)
        except ValueError: # Should not happen if logic above is correct, but safeguard
            current_selection_index = 0
            if st.session_state.available_models: # Ensure list not empty
                 st.session_state.selected_model = st.session_state.available_models[0]
            else: # This state should ideally be prevented by outer if
                 st.session_state.selected_model = None
        
        if st.session_state.selected_model is not None: # Check again after potential defaulting
            st.session_state.selected_model = st.selectbox(
                "Select Model:",
                options=st.session_state.available_models,
                index=current_selection_index,
                help="Choose from models loaded in your LM Studio instance."
            )
            st.caption(f"Currently using: `{st.session_state.selected_model}`")
    elif st.session_state.api_base_url and st.session_state.models_loaded_for_current_url and not st.session_state.models_error:
        # URL set, load attempted, no error, but no models - means API returned empty list
        st.info("No models were found at the specified LM Studio endpoint. Ensure models are loaded and served via API.")
    
    # Explicitly ensure selected_model is None if no models are available from any path
    if not st.session_state.available_models:
        st.session_state.selected_model = None

    st.markdown("---")
    st.header("💬 Chat Settings")
    st.session_state.system_prompt = st.text_area(
        "System Prompt:", value=st.session_state.system_prompt, height=100, 
        help="Initial instructions for the AI model."
    )
    st.session_state.temperature = st.slider(
        "Temperature:", min_value=0.0, max_value=2.0, value=st.session_state.temperature, step=0.05, 
        help="Controls randomness (0.0=deterministic, ~0.7=balanced, >1.0=more creative)."
    )
    # Report Section 4: Clarify max_tokens behavior
    st.session_state.max_tokens = st.number_input(
        "Max Response Tokens (0 for model default/unlimited):", 
        min_value=0, max_value=32768, value=st.session_state.max_tokens, step=64, 
        help="Max tokens for AI's response. '0' often means use model's default or effectively no limit (server-dependent)."
    )
    st.session_state.enable_streaming = st.toggle(
        "Enable Streaming Response", value=st.session_state.enable_streaming, 
        help="Receive AI's response token by token."
    )

    st.markdown("---")
    if st.button("✨ New Chat Session", help="Clears chat history and resets system prompt."):
        st.session_state.messages = []
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT # Reset system prompt
        st.success("Chat history and system prompt have been reset!")
        st.rerun() # Update UI

# --- Main Chat Area ---
# Report Section 5.3: Display historical messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input for user prompt
if prompt := st.chat_input("Ask your local AI anything..."):
    # Pre-submission checks (Report Section 5.3: Defensive check for selected_model)
    if not st.session_state.api_base_url:
        st.error("⚠️ API Base URL is not configured. Please set it in the sidebar.")
    elif not st.session_state.selected_model:
        st.error("⚠️ No LM Studio model selected. Please choose a model from the sidebar (and refresh if needed).")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        # Report Section 5.3: Construct payload with system prompt
        api_messages_payload: List[Dict[str, str]] = [{"role": "system", "content": st.session_state.system_prompt}] + \
                               [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty() # For smooth streaming display
            full_response_content: str = ""
            error_occurred: bool = False # Flag to track if an error string was the final output

            if st.session_state.enable_streaming:
                # Report Section 5.3: Use st.spinner for streaming too for consistency
                with st.spinner(f"AI ({st.session_state.selected_model}) is thinking... (streaming)"):
                    try:
                        stream_generator = generate_chat_response(
                            st.session_state.api_base_url, st.session_state.selected_model,
                            api_messages_payload, st.session_state.temperature,
                            st.session_state.max_tokens, stream=True,
                            api_key=st.session_state.api_key
                        )
                        for chunk in stream_generator:
                            if chunk.startswith("Error:"): # Check if the chunk itself is an error message
                                full_response_content = chunk 
                                error_occurred = True
                                break 
                            full_response_content += chunk
                            message_placeholder.markdown(full_response_content + "▌") # Typing cursor
                        message_placeholder.markdown(full_response_content) # Display final message or error from stream
                    except Exception as e: 
                        # Catch errors if the generator itself fails (e.g., network issue before first yield)
                        full_response_content = f"Critical error during streaming setup: {str(e)}"
                        message_placeholder.error(full_response_content)
                        error_occurred = True
            else: # Non-streaming mode
                with st.spinner(f"AI ({st.session_state.selected_model}) is thinking... (non-streaming)"):
                    ai_content, error_message_str = generate_chat_response(
                        st.session_state.api_base_url, st.session_state.selected_model,
                        api_messages_payload, st.session_state.temperature,
                        st.session_state.max_tokens, stream=False,
                        api_key=st.session_state.api_key
                    )
                if error_message_str:
                    st.error(error_message_str) # Display error directly
                    error_occurred = True
                elif ai_content:
                    st.markdown(ai_content)
                    full_response_content = ai_content
                else: # Empty but valid response
                    st.warning("AI returned an empty response (non-streaming).")
                    # Consider if an empty response should also prevent adding user message (currently doesn't pop)
                    # For now, an empty response is still a "response" from AI.
                    # If you want to treat empty as an error for popping, set error_occurred = True here.
            
            # Add assistant's response to history if it's not an error and not empty
            if not error_occurred and full_response_content:
                 st.session_state.messages.append({"role": "assistant", "content": full_response_content})
            # Removed automatic popping of user message on error to align with user feedback.
            # If desired, uncomment the following block:
            # elif error_occurred:
            #     if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            #         st.session_state.messages.pop() 
        
        # Rerun to update the displayed chat history from st.session_state.messages
        st.rerun()
