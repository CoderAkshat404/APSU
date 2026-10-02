"""APSU-RAGA core: load txt -> chunk -> TF-IDF index -> retrieve -> generate."""
import os, re, glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
FALLBACK_FILE = os.path.join(BASE, "rag", "fallback-instructions.txt")
FALLBACK_MSG = ("Ye jaankari mere paas abhi nahi hai. Kripya APSU admission office se "
                "confirm karein. Registrar Office ka number +91-9301028891 hai.")
OUT_OF_SCOPE = ("Main APSU Rewa Helpdesk Voice Agent hoon. Main APSU ke courses, fees, "
                "departments, faculty, admission aur university related information mein help kar sakta hoon.")

MIN_SCORE = float(os.getenv("MIN_SCORE", "0.12"))
TOP_K = int(os.getenv("TOP_K", "6"))
STOP = set("ka ki ke hai hain mein me se ko kya kitni kitna kitne ye wo aur ya the is of in a an to for and what how much tell about please batao bataiye".split())


SYN = {"maths": "mathematics", "math": "mathematics", "cs": "computer science", "vc": "vice chancellor",
       "fees": "fee", "kharcha": "fee", "shulk": "fee", "hod": "head", "kulpati": "vice chancellor"}


def norm(s):
    s = s.lower()
    for _ in range(3):
        s = re.sub(r"\b([a-z]+)\.(?=[a-z])", r"\1", s)       # B.Sc -> bsc, M.A. -> ma
    s = s.replace("vice-chancellor", "vice chancellor")
    return s


def tok(s):
    out = []
    for w in re.findall(r"[\w\u0900-\u097F]+", norm(s)):
        if w not in STOP:
            out.extend(SYN.get(w, w).split())
    return out


def load_chunks(max_chars=320):
    chunks = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.txt"))):
        name = os.path.basename(path)
        heading, buf = name, []

        def flush():
            if buf:
                chunks.append({"source": name, "heading": heading, "text": "\n".join(buf)})
                buf.clear()

        for line in open(path, encoding="utf-8").read().splitlines():
            line = line.strip()
            if not line or line.startswith("This file contains"):
                continue
            if line.startswith("===") or (line.isupper() and len(line) < 90):
                flush(); heading = line.strip("= ").strip(); continue
            if sum(len(x) for x in buf) + len(line) > max_chars:
                flush()
            buf.append(line)
        flush()
    return chunks


class Engine:
    def __init__(self):
        self.chunks = load_chunks()
        docs = [norm(f"{c['heading']} {c['text']}") for c in self.chunks]
        self.word = TfidfVectorizer(tokenizer=tok, lowercase=False, token_pattern=None, ngram_range=(1, 2), sublinear_tf=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.Xw, self.Xc = self.word.fit_transform(docs), self.char.fit_transform(docs)
        self.system = open(FALLBACK_FILE, encoding="utf-8").read()
        self.client = None
        if os.getenv("ANTHROPIC_API_KEY"):
            import anthropic
            self.client = anthropic.Anthropic()
            self.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    def retrieve(self, q, k=TOP_K):
        s = 0.6 * cosine_similarity(self.word.transform([norm(q)]), self.Xw)[0] \
          + 0.4 * cosine_similarity(self.char.transform([norm(q)]), self.Xc)[0]
        idx = np.argsort(-s)[:k]
        return [dict(self.chunks[i], score=float(s[i])) for i in idx]

    def extractive(self, q, hits):
        qt = set(tok(q)); idf = dict(zip(self.word.get_feature_names_out(), self.word.idf_)); lines = []
        for h in hits:
            ht = set(tok(h["heading"]))
            for ln in h["text"].splitlines():
                lt = set(tok(ln)) | ht
                lines.append((sum(idf.get(w, 1.0) for w in qt & lt), ln))
        lines.sort(key=lambda x: -x[0])
        if not lines or lines[0][0] == 0:
            return hits[0]["text"][:400]
        top = lines[0][0]
        out = []
        for sc, ln in lines:
            if sc >= 0.85 * top and ln not in out:
                out.append(ln)
            if len(out) == 3:
                break
        return "\n".join(out)

    def answer(self, q, history=None):
        hits = self.retrieve(q, 2 * TOP_K)
        if not hits or hits[0]["score"] < MIN_SCORE:
            on_topic = re.search(r"apsu|rewa|university|college|course|fee|admission|vc |registrar|hostel|syllabus", q, re.I)
            return {"answer": FALLBACK_MSG if on_topic else OUT_OF_SCOPE, "sources": [], "mode": "fallback"}
        sources = [{"file": h["source"], "section": h["heading"], "score": round(h["score"], 2)} for h in hits[:TOP_K]]
        if not self.client:
            return {"answer": self.extractive(q, hits), "sources": sources, "mode": "extractive"}
        ctx = "\n\n".join(f"[{h['source']} | {h['heading']}]\n{h['text']}" for h in hits[:TOP_K])
        msgs = [{"role": m["role"], "content": m["content"]} for m in (history or [])[-6:]]
        msgs.append({"role": "user", "content": f"KNOWLEDGE BASE CONTEXT:\n{ctx}\n\nQUESTION: {q}\n\nAnswer ONLY from the context, in the user's language (Hindi/Hinglish/English), short and direct."})
        try:
            r = self.client.messages.create(model=self.model, max_tokens=500, system=self.system, messages=msgs)
            return {"answer": r.content[0].text.strip(), "sources": sources, "mode": "rag-llm"}
        except Exception as e:
            return {"answer": self.extractive(q, hits), "sources": sources, "mode": f"extractive (LLM error: {type(e).__name__})"}
