import os
from typing import List, Dict

SYSTEM_PROMPT = """You are an RFP assistant. Answer ONLY with the provided context.
If the answer is missing from the context, say you cannot find it.
Focus on deadlines, budget, scope, eligibility, deliverables when relevant.
Answer in Korean unless the user asks otherwise.
"""

def _openai_answer(query: str, contexts: List[Dict]) -> str:
    from openai import OpenAI
    client = OpenAI()
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    ctx = "\n\n".join([f"[p{c.get('page','?')}|{c.get('doc_id','?')}] {c['text']}" for c in contexts])
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}"}
        ],
        temperature=float(os.getenv("GEN_TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("GEN_MAX_TOKENS", "512")),
    )
    return resp.choices[0].message.content

def generate_answer(query: str, contexts: List[Dict]) -> str:
    provider = os.getenv("GEN_PROVIDER", "openai")
    if provider == "openai":
        return _openai_answer(query, contexts)
    raise NotImplementedError("Only OpenAI provider is implemented.")
