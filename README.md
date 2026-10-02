# APSU-RAGA — Retrieval-Augmented Generative Assistant for A.P.S. University, Rewa
Chatbot (text + voice) that answers ONLY from the university's `.txt` knowledge base.

## Folder structure
```
apsu-raga/
├── app.py                  # Flask server (API + serves frontend)
├── requirements.txt
├── .env.example            # copy to .env (optional LLM key)
├── data/                   # knowledge base (add/edit .txt files here)
│   ├── general-info.txt  courses-list.txt  fee-structure.txt
│   ├── admission-process.txt  department-details.txt
│   ├── university-administration.txt  syllabus-detailed.txt
├── rag/
│   ├── engine.py           # chunking, TF-IDF retrieval, generation, fallback logic
│   └── fallback-instructions.txt   # system prompt / no-hallucination rules
└── static/index.html       # chat UI (voice input, read-aloud, sources)
```

## Run (Windows / Linux / Mac)
1. Install Python 3.9+.
2. `cd apsu-raga`
3. `python -m venv venv` then activate: Windows `venv\Scripts\activate` / Linux-Mac `source venv/bin/activate`
4. `pip install -r requirements.txt`
5. (Optional, better answers) `copy .env.example .env` (Linux: `cp`) and put your `ANTHROPIC_API_KEY` in it.
6. `python app.py`
7. Open **http://127.0.0.1:5000** in Chrome/Edge (voice input needs Chrome/Edge).

Without an API key it runs in **extractive mode** (returns the best-matching lines from the txt files, fully offline). With a key it runs full **RAG**: retrieved chunks are given to the LLM with the fallback rules.

## How it works (for viva)
txt files → cleaned & chunked (by section heading, ≤320 chars) → TF-IDF index (word 1–2-grams + character 3–5-grams, so Hinglish spelling variations still match) → query normalised (synonyms: maths→mathematics, VC→vice chancellor) → top-6 chunks by cosine similarity → if best score < `MIN_SCORE` return the fallback message (anti-hallucination) → else LLM (or extractive) answers from context only → sources shown under each answer.

## Tuning
`MIN_SCORE` in `.env` (higher = stricter, more fallbacks). Add new knowledge by dropping a `.txt` file in `data/` and restarting.
Voice: browser Web Speech API (STT `hi-IN`, TTS toggle in footer). 
