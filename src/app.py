import os, json, time, datetime as dt
import streamlit as st

from retriever_tfidf import TfidfRetriever
from retriever_vect import EmbeddingRetriever
from llm_local import LocalLLM
from llm_api import OpenAILLM

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "history.jsonl")

st.set_page_config(page_title="Mini LLM Chatbot (RAG)", page_icon="🧠")
st.title("🧠 Mini LLM Chatbot — RAG + Streaming + Embeddings + Логи")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Настройки")
    backend = st.selectbox("Backend", ["Local (HF)", "API (OpenAI)"], index=0)
    retriever_kind = st.selectbox("Retriever", ["TF-IDF", "Embeddings (FAISS)"], index=0)
    topk_ctx = st.slider("Top‑K контекстов", 0, 10, 3, 1)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    enable_stream = st.checkbox("Streaming (только API)", value=True)
    enable_log = st.checkbox("Сохранять логи (JSONL)", value=True)

    if backend.startswith("Local"):
        model_name = st.text_input("HF модель", "Qwen/Qwen2.5-0.5B-Instruct")
        max_tokens = st.slider("Max new tokens (Local)", 64, 1024, 256, 32)
    else:
        api_model = st.text_input("OpenAI модель", "gpt-4o-mini")
        max_tokens = st.slider("Max output tokens (API)", 64, 2048, 400, 32)
        st.caption("Для API режима укажи OPENAI_API_KEY в .env")

@st.cache_resource(show_spinner=True)
def bootstrap_retriever(kind: str):
    if kind == "TF-IDF":
        return TfidfRetriever("data/faq/*.md", chunk_size=600, overlap=150)
    else:
        return EmbeddingRetriever("data/faq/*.md", model_name="sentence-transformers/all-MiniLM-L6-v2", chunk_size=600, overlap=150)

@st.cache_resource(show_spinner=True)
def bootstrap_local(model_name: str, max_tokens: int):
    return LocalLLM(model_name=model_name, max_new_tokens=max_tokens)

@st.cache_resource(show_spinner=True)
def bootstrap_api(api_model: str, max_tokens: int):
    return OpenAILLM(model=api_model, max_output_tokens=max_tokens)

# показать историю
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

system_prompt = st.text_area(
    "Системный промпт",
    "Ты — дружелюбный и краткий помощник. Отвечай по существу, пиши на русском. "
    "Если контент не найден в документах — отвечай на общих знаниях и помечай это явно."
)

query = st.chat_input("Задай вопрос…")
if query:
    st.session_state.messages.append({"role": "user", "content": query})

    retr = bootstrap_retriever(retriever_kind)

    # контекст из ретривера
    context_blocks = ""
    retrieved = []
    if topk_ctx > 0:
        retrieved = retr.topk(query, k=topk_ctx)
        context_blocks = "\n\n".join([f"[Контекст {i+1} | {meta} | score={score:.3f}]\n{chunk}" for i, (chunk, meta, score) in enumerate(retrieved)])

    history = st.session_state.messages[-6:]

    t0 = time.time()
    if backend.startswith("Local"):
        llm = bootstrap_local(model_name, max_tokens)
        with st.spinner("Генерирую ответ (Local)…"):
            answer = llm.chat(system=system_prompt, messages=history, context_blocks=context_blocks, temperature=temperature)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
    else:
        llm = bootstrap_api(api_model, max_tokens)
        if enable_stream:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                chunks = []
                for delta in llm.chat_stream(system=system_prompt, messages=history, context_blocks=context_blocks, temperature=temperature):
                    chunks.append(delta)
                    placeholder.markdown("".join(chunks))
                answer = "".join(chunks)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            with st.spinner("Генерирую ответ (API)…"):
                answer = llm.chat(system=system_prompt, messages=history, context_blocks=context_blocks, temperature=temperature)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)

    t1 = time.time()

    # ЛОГИ
    if enable_log:
        log_rec = {
            "ts": dt.datetime.utcnow().isoformat() + "Z",
            "backend": "local" if backend.startswith("Local") else "api",
            "model": model_name if backend.startswith("Local") else api_model,
            "retriever": "tfidf" if retriever_kind == "TF-IDF" else "embeddings",
            "query": query,
            "answer_preview": answer[:200],
            "ctx_used": len(retrieved),
            "elapsed_sec": round(t1 - t0, 3),
        }
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_rec, ensure_ascii=False) + "\n")

st.caption("Переключай бэкенд/ретривер в сайдбаре. В API‑режиме доступен стриминг. Логи — в logs/history.jsonl.")
