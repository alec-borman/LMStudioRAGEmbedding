
# LM Studio RAG Chat App 🤖

A **local Retrieval-Augmented Generation (RAG) chat application** built with [Streamlit](https://streamlit.io) and [LM Studio](https://lmstudio.ai/). This app uses two GGUF models:
- `Qwen3-14B` (`qwen3-14b-128K-Q4_K_M.gguf`) for **generative chat responses**
- `Nomic Embed Text-v1.5` (`nomic-embed-text-v1.5.Q8_0.gguf`) for **semantic search and embedding-based contextual augmentation**

---

## 🚀 Overview

This project enables developers to run a secure, cost-effective, and context-aware AI chat interface entirely on local hardware using open-source tools. It eliminates reliance on cloud APIs while providing dynamic conversation history integration via RAG.

**Key Features:**
- **Privacy First:** All data stays on your machine
- **Dual Model Architecture:** Generative model + Embedding model for enhanced accuracy
- **RAG Implementation:** Uses cosine similarity to retrieve relevant past messages
- **Streaming Support:** Real-time response rendering with `st.write_stream`
- **Configurable Parameters:** Temperature, max tokens, and RAG context depth
- **Robust Error Handling:** Comprehensive diagnostics for API communication

---

## 🧩 Architecture Overview

| Component | Role |
|----------|------|
| **Streamlit** | Interactive UI for chat input/output and model configuration |
| **LM Studio Server** | Local LLM server that exposes models via an OpenAI-compatible REST API (`http://localhost:1234/v1`) |
| **Qwen3-14B (GGUF)** | Main model used for conversational AI with extended context support |
| **Nomic Embed Text-v1.5 (GGUF)** | Model used to generate embeddings for contextual relevance search |
| **RAG Logic** | Dynamically retrieves and augments conversation history in prompts |

---

## 📦 Requirements

### Python & Dependencies
```bash
Python 3.10+
pip install streamlit requests numpy scikit-learn
```

### LM Studio
- Installed from [LM Studio GitHub](https://github.com/jondurbin/lmstudio)
- Models must be placed in the default directory:
  ```plaintext
  C:\Users\<YourUsername>\.cache\lm-studio\models\
  ```
- Model filenames **must match exactly** as defined in the script (see `model_id` constants).

---

## 🧰 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/lm_studio_rag_chat.git
cd lm_studio_rag_chat
```

### 2. Create & Activate Virtual Environment
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# or
venv\Scripts\activate.bat    # Windows CMD
```

> 💡 For PowerShell users, if activation fails:
```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Install Dependencies
```bash
pip install streamlit requests numpy scikit-learn --upgrade
```

---

## 📥 Required Models

| Model ID Constant | GGUF Filename | Download Source |
|-------------------|---------------|------------------|
| `QWEN3_CHAT_MODEL_ID` | `qwen3-14b-128K-Q4_K_M.gguf` | [Hugging Face - MaziyarGudarzi/Qwen1.5-14B-Chat-GGUF](https://huggingface.co/MaziyarGudarzi/Qwen1.5-14B-Chat-GGUF) |
| `NOMIC_EMBEDDING_MODEL_ID` | `nomic-embed-text-v1.5.Q8_0.gguf` | [Hugging Face - nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) |

> ⚠️ **Model Filename Matching is Critical**  
The script uses these exact filenames as model IDs through LM Studio's API. Ensure the GGUF files are named correctly and placed in a directory LM Studio can access (e.g., its default cache folder or a custom path defined in settings).

---

## 🧪 Verify LM Studio API Access

Ensure the server is running and models are available via the `/v1/models` endpoint.

### Test with PowerShell:
```powershell
Invoke-WebRequest -Uri "http://localhost:1234/v1/models"
```

### Expected JSON Response:
```json
{
  "object": "list",
  "data": [
    {"id": "qwen3-14b-128K-Q4_K_M.gguf"},
    {"id": "nomic-embed-text-v1.5.Q8_0.gguf"}
  ]
}
```

---

## 🚁 Run the Chat Application

### Launch via Terminal
```bash
streamlit run lm_studio_chat.py
```

- A new browser tab will open at `http://localhost:8501`
- The sidebar lets you:
  - Configure the LM Studio API endpoint (default: `http://localhost:1234/v1`)
  - Toggle streaming for real-time response rendering
  - Adjust temperature and max tokens

---

## 📌 Technical Details

### Key Model Roles in the Script
| Constant | GGUF Filename | Role |
|----------|---------------|------|
| `QWEN3_CHAT_MODEL_ID` | `qwen3-14b-128K-Q4_K_M.gguf` | Primary LLM for chat generation |
| `NOMIC_EMBEDDING_MODEL_ID` | `nomic-embed-text-v1.5.Q8_0.gguf` | Embedding model for semantic search in RAG |

---

## 🔍 Security & Performance Considerations

### 🛡️ Security
- **GGUF Models:** The GGML library has known vulnerabilities (e.g., CVE-2024-25664). Ensure you use the latest patched version of LM Studio.
- **Model Sources:** Only download from trusted repositories (Hugging Face, official model maintainers).

### 📈 Performance
- **VRAM Usage:** `qwen3-14b-128K-Q4_K_M.gguf` uses ~9.1 GB. Adjust quantization in LM Studio if you experience performance issues.
- **Caching:** Embeddings and model availability are cached using `@st.cache_data` to optimize repeated calls.

---

## 🧪 Troubleshooting

| Issue | Solution |
|------|----------|
| `ModuleNotFoundError: 'streamlit'` | Activate virtual environment, reinstall dependencies with `pip install --upgrade` |
| "Models NOT FOUND" in UI | Ensure model filenames match script constants; refresh LM Studio's models list |
| API connection errors ("Connection refused") | Start LM Studio server and verify port 1234 is open (use `netstat -ano | findstr :1234`) |
| JSON parsing errors (`json.JSONDecodeError`) | Check LM Studio logs for malformed responses; ensure the model is correctly loaded |

---

## 📌 Contribution & Roadmap

### 🔧 How to Contribute
- Report bugs via [GitHub Issues](https://github.com/yourusername/lm_studio_rag_chat/issues)
- Feature requests in the same GitHub Issues tracker
- Pull requests for model integration, documentation, or performance optimizations

---

## 📜 License

This project is licensed under the **MIT License** – see [LICENSE](LICENSE) file for details.

---

## 💡 Notes on Model Usage & RAG Process

1. **User Input:** Captured via Streamlit UI
2. **Embed Query:** Nomic model generates a vector of semantic meaning from the input
3. **RAG Search:** Cosine similarity finds most relevant past messages using their embeddings
4. **Prompt Augmentation:** Combined with system prompt and user query for Qwen3 processing
5. **Streamlit Rendering:** Response is shown in real-time if streaming enabled

---

## 🧰 Additional Tools & Resources

### LM Studio Setup:
- Visit [LM Studio Docs](https://lmstudio.ai/)
- Ensure you have **AVX2-compatible hardware** for optimal performance on Windows/Linux.

### RAG Understanding:
- [Google Cloud RAG Guide](https://cloud.google.com/ai/rag)
- [AWS RAG Documentation](https://aws.amazon.com/what-is-rag/)

---

## 📌 Example Usage

1. Start LM Studio and load both models
2. Run the app with `streamlit run lm_studio_chat.py`
3. Click "Load Models from API" in the sidebar to verify model availability
4. Adjust settings (temperature, max tokens) and enable streaming for a dynamic experience
5. Type your question: `What is Retrieval-Augmented Generation?`

> 💡 Your response will be generated using Qwen3-14B with RAG context retrieved via Nomic Embed.

---

## 📌 Want to Containerize This App?

Consider creating a Dockerfile to make the app portable and easy to deploy on any system. If you'd like help writing one, just ask!

---

## 🔗 References

- [Streamlit Documentation](https://docs.streamlit.io)
- [LM Studio GitHub Repository](https://github.com/jondurbin/lmstudio)
- [Qwen3 Model Info - Toolify AI](https://toolify.ai/qwen3)
- [Nomic Embed Text-v1.5 - Hugging Face](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)

---

## 📝 Final Notes

This RAG chat app is designed to be **user-friendly, efficient, and secure**, making it an excellent choice for developers who want local control over AI systems. With continued maintenance of the underlying LM Studio and GGML libraries, this project remains a powerful and safe alternative to cloud-based solutions.
