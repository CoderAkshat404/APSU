from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify, send_from_directory
from rag.engine import Engine
import os

app = Flask(__name__, static_folder="static")
engine = Engine()
print(f"Indexed {len(engine.chunks)} chunks | LLM: {'on' if engine.client else 'off (extractive mode)'}")

@app.get("/")
def home():
    return send_from_directory("static", "index.html")

@app.post("/api/chat")
def chat():
    d = request.get_json(force=True)
    q = (d.get("message") or "").strip()
    if not q:
        return jsonify({"error": "Empty message"}), 400
    return jsonify(engine.answer(q, d.get("history")))

@app.get("/api/status")
def status():
    return jsonify({"chunks": len(engine.chunks), "llm": bool(engine.client)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
