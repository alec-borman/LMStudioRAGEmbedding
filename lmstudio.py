Pythonimport streamlit as st
import requests
import json

# --- App Configuration ---
LM_STUDIO_DEFAULT_URL = "http://localhost:1234/v1/chat/completions"
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."

# --- API Communication Logic ---

def get_available_models(api_url):
    """Fetches the list of available models from the LM Studio API."""
    models_endpoint = api_url.replace("/chat/completions", "/models")
    try:
        response = requests.get(models_endpoint, timeout=10)
        response.raise_for_status()
        models_data = response.json()

        if isinstance(models_data, dict) and "data" in models_data and isinstance(models_data["data"], list):
            # Standard OpenAI API structure
            model_ids = [model.get("id") for model in models_data["data"] if model.get("id")]
            return model_ids, None
        elif isinstance(models_data, list): # Fallback if the root is a list of models
            model_ids = [model.get("id") for model in models_data if isinstance(model, dict) and model.get("id")]
            return model_ids, None
        else:
            # Attempt to find a list of models if structured differently (e.g. LM Studio specific)
            # This part might need adjustment based on actual LM Studio /v1/models raw response
            for key, value in models_data.items():
                if isinstance(value, list) and value and isinstance(value, dict) and "id" in value:
                    model_ids = [model.get("id") for model in value if model.get("id")]
                    return model_ids, None
            # If still not found, check for LM Studio SDK-like structure (model_key)
            if isinstance(models_data, dict) and "data" in models_data and isinstance(models_data["data"], list):
                 model_ids = [model.get("model_key") for model in models_data["data"] if model.get("model_key")]
                 if model_ids:
                     return model_ids, None
            
            st.warning(f"Unexpected response structure from {models_endpoint}. Could not extract model list. Raw: {str(models_data)[:200]}")
            return, f"Unexpected response structure from /models."

    except requests.exceptions.Timeout:
        return, "Error: Timeout when fetching models."
    except requests.exceptions.RequestException as e:
        return, f"Error fetching models: {str(e)}"
    except (json.JSONDecodeError, KeyError) as e:
        raw_response_text = response.text if 'response' in locals() else 'N/A'
        return, f"Error parsing models response: {str(e)}. Raw: {raw_response_text[:200]}"
    return, "Could not retrieve models. Ensure LM Studio is running and a model is loaded."


def get_lm_studio_response_stream(api_url, model_id, messages_payload, temperature, max_tokens):
    """
    Yields response content from LM Studio API using streaming.
    Handles OpenAI-compatible SSE format.
    """
    payload = {
        "model": model_id,
        "messages": messages_payload,
        "temperature": temperature,
        "max_tokens": max_tokens if max_tokens > 0 else None,  # None for unlimited, or specific number
        "stream": True
    }
    
    with requests.post(api_url, json=payload, stream=True, timeout=300) as response: # Increased timeout for potentially long streams
        response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    json_data_str = decoded_line[len('data: '):].strip()
                    if json_data_str == "":
                        break
                    if json_data_str:
                        try:
                            chunk = json.loads(json_data_str)
                            if chunk.get("choices") and len(chunk["choices"]) > 0:
                                delta = chunk["choices"].get("delta", {})
                                content_piece = delta.get("content")
                                if content_piece:
                                    yield content_piece
                        except json.JSONDecodeError:
                            # LM Studio might send other SSE events or malformed JSON, skip them
                            # print(f"Skipping non-JSON or malformed data line: {json_data_str}")
                            pass 


def get_lm_studio_response_nostream(api_url, model_id, messages_payload, temperature, max_tokens):
    """Gets a non-streaming response from LM Studio API."""
    payload = {
        "model": model_id,
        "messages": messages_payload,
        "temperature": temperature,
        "max_tokens": max_tokens if max_tokens > 0 else None,
        "stream": False
    }
    try:
        response = requests.post(api_url, json=payload, timeout=120)
        response.raise_for_status()
        assistant_response = response.json()
        content = assistant_response.get("choices", [{}]).get("message", {}).get("content", "")
        return content, None 
    except requests.exceptions.Timeout:
        return None, "Error: The request to LM Studio timed out."
    except requests.exceptions.RequestException as e:
        return None, f"Error connecting to LM Studio: {str(e)}"
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        raw_response_text = response.text if 'response' in locals() else 'N/A'
        return None, f"Error parsing response from LM Studio: {str(e)}. Raw: {raw_response_text[:200]}"


# --- Streamlit UI ---
st.set_page_config(page_title="LMChat Studio Interface", layout="wide")
st.title("LMChat Studio Interface 🤖")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages =
if "api_url" not in st.session_state:
    st.session_state.api_url = LM_STUDIO_DEFAULT_URL
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 512 # Set to 0 or -1 in LM Studio for unlimited
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "available_models" not in st.session_state:
    st.session_state.available_models =
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "enable_streaming" not in st.session_state:
    st.session_state.enable_streaming = True


# --- Sidebar for Settings ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.session_state.api_url = st.text_input("LM Studio API Endpoint:", value=st.session_state.api_url)

    # Load models if API URL is provided and models haven't been loaded yet or URL changed
    # This is a simple way to trigger reload; more sophisticated checks could be added
    if st.button("Load Models from API") or (st.session_state.api_url and not st.session_state.models_loaded):
        with st.spinner("Fetching available models..."):
            models, error = get_available_models(st.session_state.api_url)
            if error:
                st.error(f"Could not load models: {error}")
                st.session_state.available_models =
                st.session_state.selected_model = None
            else:
                if models:
                    st.session_state.available_models = models
                    if st.session_state.selected_model not in models:
                        st.session_state.selected_model = models
                    st.success("Models loaded successfully!")
                else:
                    st.warning("No models found or returned by the API.")
                    st.session_state.available_models =
                    st.session_state.selected_model = None
            st.session_state.models_loaded = True # Mark as loaded to prevent re-fetching on every rerun unless button is pressed
            st.rerun() # Rerun to update the selectbox with new models

    if st.session_state.available_models:
        # Ensure the current selected_model is valid, otherwise default to the first available
        current_selection_index = 0
        if st.session_state.selected_model in st.session_state.available_models:
            current_selection_index = st.session_state.available_models.index(st.session_state.selected_model)
        elif st.session_state.available_models: # Default to first if current is invalid but list is not empty
             st.session_state.selected_model = st.session_state.available_models

        st.session_state.selected_model = st.selectbox(
            "Select Model:",
            options=st.session_state.available_models,
            index=current_selection_index
        )
    else:
        st.info("Click 'Load Models from API' or ensure LM Studio server is running with models loaded.")

    st.session_state.system_prompt = st.text_area("System Prompt:", value=st.session_state.system_prompt, height=100)
    st.session_state.temperature = st.slider("Temperature:", min_value=0.0, max_value=2.0, value=st.session_state.temperature, step=0.05)
    st.session_state.max_tokens = st.number_input("Max Tokens (0 for unlimited):", min_value=0, max_value=16384, value=st.session_state.max_tokens, step=64)
    st.session_state.enable_streaming = st.toggle("Enable Streaming Response", value=st.session_state.enable_streaming)

    if st.button("✨ New Chat"):
        st.session_state.messages =
        st.rerun()

    st.caption(f"LM Studio API: {st.session_state.api_url}")
    if st.session_state.selected_model:
        st.caption(f"Selected Model: {st.session_state.selected_model}")

    if st.button("Test Connection to /models"):
        models_test_url = st.session_state.api_url.replace("/chat/completions", "/models")
        try:
            test_response = requests.get(models_test_url, timeout=5)
            if test_response.status_code == 200:
                st.success(f"Successfully connected to {models_test_url}!")
                try:
                    st.json(test_response.json(), expanded=False)
                except json.JSONDecodeError:
                    st.info("Response was not JSON, but connection was successful.")
                    st.text(test_response.text[:500])
            else:
                st.warning(f"Connected to {models_test_url}, but received status: {test_response.status_code}\nResponse: {test_response.text[:200]}")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to {models_test_url}: {e}")

# --- Main Chat Area ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to ask?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not st.session_state.selected_model:
        st.error("Please select a model from the sidebar first.")
    else:
        api_messages_payload = [{"role": "system", "content": st.session_state.system_prompt}] + st.session_state.messages
        
        with st.chat_message("assistant"):
            if st.session_state.enable_streaming:
                try:
                    stream_generator = get_lm_studio_response_stream(
                        st.session_state.api_url,
                        st.session_state.selected_model,
                        api_messages_payload,
                        st.session_state.temperature,
                        st.session_state.max_tokens
                    )
                    # st.write_stream accumulates the response and returns the full string
                    full_response_content = st.write_stream(stream_generator)
                    
                    # Check if the stream itself yielded an error message (less ideal, but a fallback)
                    if isinstance(full_response_content, str) and full_response_content.startswith("Error:"):
                        st.error(full_response_content) # Display error within the assistant message block
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": full_response_content})
                
                except requests.exceptions.RequestException as e:
                    st.error(f"API Request Error: {str(e)}")
                except Exception as e: # Catch other errors during streaming setup or processing by st.write_stream
                    st.error(f"An unexpected error occurred: {str(e)}")
            else: # Non-streaming
                with st.spinner("AI is thinking..."):
                    ai_content, error_message = get_lm_studio_response_nostream(
                        st.session_state.api_url,
                        st.session_state.selected_model,
                        api_messages_payload,
                        st.session_state.temperature,
                        st.session_state.max_tokens
                    )
                if error_message:
                    st.error(error_message)
                elif ai_content:
                    st.markdown(ai_content)
                    st.session_state.messages.append({"role": "assistant", "content": ai_content})
                else:
                    st.error("AI returned an empty response.")
