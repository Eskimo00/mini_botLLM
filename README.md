# 🧠 Mini LLM Chatbot (локально или через API) — RAG + Streaming + Embeddings + Логи

Готовый чатбот на Streamlit с двумя режимами бэкенда и двумя типами ретривера:
- **Бэкенды**: Local (HuggingFace) / API (OpenAI)
- **Ретриверы**: TF‑IDF / Embeddings (FAISS, sentence‑transformers)

Добавлено:
- ✅ **Стриминг** ответов в API‑режиме (галочка в сайдбаре)
- ✅ **Embeddings‑ретривер** на FAISS (`sentence-transformers/all-MiniLM-L6-v2`)
- ✅ **Логирование** запросов/ответов в `logs/history.jsonl`

## Установка
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt

# данные для RAG
mkdir -p data/faq
echo "# FAQ
Это пример документа." > data/faq/example.md

# (опционально для API)
cp .env.example .env
# открой .env и укажи OPENAI_API_KEY=sk-...
```

## Запуск
```bash
streamlit run src/app.py
```
В сайдбаре выберите Backend и Retriever, при желании включите Streaming и Logging.

## Структура
```
mini-llm-chatbot/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   └── faq/
│       └── .gitkeep
├── logs/
│   └── .gitkeep
└── src/
    ├── app.py
    ├── retriever_tfidf.py
    ├── retriever_vect.py
    ├── llm_local.py
    └── llm_api.py
```
