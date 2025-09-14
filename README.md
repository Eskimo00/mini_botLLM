# 🧠 Mini LLM Chatbot — Retrieval-Augmented Generation

Мини-чатбот на базе LLM с поддержкой поиска по документам (RAG).  
Демонстрирует навыки интеграции локальных и API-моделей, построения поиска по документам и работы со стримингом в интерфейсе **Streamlit**.

---

## 🚀 Возможности
- **Два режима работы**:
  - Local (HF, через `transformers`)
  - API (OpenAI GPT-4o-mini или другие модели)
- **Ретриверы**:
  - TF-IDF (поиск по ключевым словам)
  - Embeddings + FAISS (поиск по смыслу)
- **Стриминг** (API-режим) — вывод ответа по мере генерации.  
- **Логирование** — каждый запрос сохраняется в `logs/history.jsonl`.  
- **Документы для поиска** — кладутся в папку `data/faq/` (формат `.md` или `.txt`).  

---

## 🛠️ Установка и запуск
```bash
git clone https://github.com/<your_username>/mini-llm-chatbot.git
cd mini-llm-chatbot

python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

(для API-режима)  
Скопируйте `.env.example` → `.env` и укажите свой ключ:
```bash
OPENAI_API_KEY=sk-...
```

Запуск:
```bash
streamlit run src/app.py
```

Интерфейс откроется на http://localhost:8501.  

---

## 📂 Пример документа для поиска
Создайте файл `data/faq/test_faq.md`:
```markdown
# FAQ
OpenAI была основана в 2015 году.
Главный офис — Сан-Франциско.
```

Задайте вопрос в чате:  
👉 «Когда была основана OpenAI?»  

---

## 📊 Что демонстрирует проект
- Работа с локальными и API-моделями (**HuggingFace, OpenAI**).  
- Построение **RAG-системы** (поиск + генерация ответа).  
- Использование эмбеддингов и **FAISS** для поиска по смыслу.  
- Визуализация через **Streamlit**.  
- Логирование запросов для анализа.
