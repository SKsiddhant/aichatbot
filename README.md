# AI Chatbots

Two small chatbots live in this repo.

## 1. Intent-based chatbot (PyTorch) — repo root

A bag-of-words neural network that classifies a message into an intent from `intents.json` and replies with a canned response. Based on the Python Engineer tutorial; see `LICENSE`.

```bash
python3 -m venv venv && . venv/bin/activate
pip install -r requirements.txt
python train.py      # writes data.pth
python chat.py       # console chat, type "quit" to exit
```

`chat.py` raises a clear error if `data.pth` is missing. NLTK's `punkt_tab` data is optional (`python -c "import nltk; nltk.download('punkt_tab')"`); without it a data-free tokenizer is used. For a Flask web UI built on the same model see the `aichatbot1` repo.

## 2. Dynamic LLM chatbot — `dynamic_chatbot/`

A `ConversationManager` class that talks to an OpenAI-compatible API (Together AI by default), supports personas, keeps conversation history on disk and trims it to a token budget.

```bash
cd dynamic_chatbot
pip install -r requirements.txt
cp .env.example .env     # add TOGETHER_AI_API_KEY
python -c "from dynamic_ai_chatbot import ConversationManager as C; b=C(); print(b.chat_completion('Hi!'))"
```

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```
