"""LangGraph RAG workflow: retrieve -> generate -> grade (groundedness)."""
import os
import re
from typing import List, TypedDict

import requests

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import END, START, StateGraph

from src import config

# --------------------------------------------------------------------------
# LLM access with automatic model discovery + fallback.
# Free-tier quotas are per model per day and model names change often, so we
# (1) try Groq first when GROQ_API_KEY is set, (2) then every Gemini model that
# the user's own key can actually call.
# --------------------------------------------------------------------------
_BAD_WORDS = ("embed", "image", "tts", "live", "audio", "imagen", "veo", "vision",
              "whisper", "guard", "playai", "transcribe", "robotics", "computer")
_failed: set = set()


def _groq_models() -> List[str]:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return []
    preferred = os.getenv("GROQ_MODEL")
    ids: List[str] = []
    try:
        r = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {key}"}, timeout=20,
        )
        ids = [m["id"] for m in r.json().get("data", [])]
    except Exception as e:  # noqa: BLE001
        print(f"[groq model list failed: {type(e).__name__}]")
    ids = [i for i in ids if not any(w in i.lower() for w in _BAD_WORDS)]
    ids.sort(key=lambda i: (0 if "llama" in i and "70b" in i else 1 if "llama" in i else 2 if "gpt-oss" in i else 3, i))
    return ([preferred] if preferred else []) + [i for i in ids if i != preferred]


def _gemini_models() -> List[str]:
    names: List[str] = []
    try:
        from google import genai

        client = genai.Client()
        for m in client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            n = (m.name or "").replace("models/", "")
            if "generateContent" in actions and "gemini" in n and not any(w in n for w in _BAD_WORDS):
                names.append(n)
    except Exception as e:  # noqa: BLE001
        print(f"[gemini model list failed: {type(e).__name__}]")
    names.sort(key=lambda n: (0 if "lite" in n else 1 if "flash" in n else 2, n))
    return [config.LLM_MODEL] + [n for n in names if n != config.LLM_MODEL]


def _call_groq(model: str, prompt: str) -> str:
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
        json={"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _call_gemini(model: str, prompt: str) -> str:
    llm = ChatGoogleGenerativeAI(model=model, temperature=0, max_retries=1)
    return _text(llm.invoke(prompt).content).strip()


_CANDIDATES: List[tuple] = []


def call_llm(prompt: str) -> str:
    """Try every available model (Groq, then Gemini) until one answers."""
    global _CANDIDATES
    if not _CANDIDATES:
        _CANDIDATES = [("groq", m) for m in _groq_models()] + [("gemini", m) for m in _gemini_models()]
    last_error = None
    for provider, name in _CANDIDATES:
        if (provider, name) in _failed:
            continue
        try:
            out = _call_groq(name, prompt) if provider == "groq" else _call_gemini(name, prompt)
            if out:
                return out
        except Exception as e:  # noqa: BLE001
            last_error = e
            _failed.add((provider, name))
            print(f"[{provider}:{name} failed: {type(e).__name__}] trying next...")
    raise RuntimeError(f"No LLM available (all models failed). Last error: {last_error}")


class AgentState(TypedDict):
    question: str
    context: List[str]
    pages: List[int]
    relevance: float   # best cosine similarity from Pinecone
    answer: str
    refused: bool
    grounded: float    # model's self-reported groundedness, 0..1
    score: float       # final confidence score


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def build_rag_graph(index_name: str = config.PINECONE_INDEX_NAME):
    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    # ---- Node 1: retrieve -------------------------------------------------
    def retrieve_node(state: AgentState):
        results = vectorstore.similarity_search_with_score(state["question"], k=config.TOP_K)
        context = [doc.page_content for doc, _ in results]
        pages = [int(doc.metadata.get("page", 0)) for doc, _ in results]
        relevance = max((float(s) for _, s in results), default=0.0)
        return {"context": context, "pages": pages, "relevance": relevance}

    # ---- Node 2: generate (grounded ONLY on retrieved context) -------------
    # One LLM call returns both the answer and a groundedness self-check.
    def generate_node(state: AgentState):
        if not state["context"] or state["relevance"] < config.MIN_RELEVANCE:
            return {"answer": config.REFUSAL, "refused": True, "grounded": 0.0, "score": 0.0}

        context_str = "\n\n---\n\n".join(state["context"])
        prompt = f"""You are a strict assistant answering ONLY from the context below.
Rules:
- Use only facts present in the context. Do not use outside knowledge.
- If the context does not contain the answer, the answer must be exactly: {config.REFUSAL}
- Be concise and well structured.

Context:
{context_str}

Question: {state['question']}

Reply in exactly this format:
ANSWER: <your answer>
GROUNDED: <a number from 0 to 1 = how fully every claim in your answer is supported by the context>"""
        raw = call_llm(prompt)

        grounded = 0.5
        answer = raw
        m = re.search(r"GROUNDED:\s*([01](?:\.\d+)?)", raw)
        if m:
            grounded = max(0.0, min(1.0, float(m.group(1))))
            answer = raw[: m.start()]
        answer = re.sub(r"^\s*ANSWER:\s*", "", answer).strip()

        refused = config.REFUSAL.lower() in answer.lower()
        return {"answer": answer, "refused": refused, "grounded": 0.0 if refused else grounded}

    # ---- Node 3: grade -> final confidence score ---------------------------
    def grade_node(state: AgentState):
        relevance = max(0.0, min(1.0, state["relevance"]))
        score = round(0.4 * relevance + 0.6 * state["grounded"], 2)
        return {"score": score}

    def route_after_generate(state: AgentState):
        return END if state["refused"] else "grade"

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("grade", grade_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_conditional_edges("generate", route_after_generate, {"grade": "grade", END: END})
    workflow.add_edge("grade", END)
    return workflow.compile()


def ask(graph, query: str) -> dict:
    """Run the graph and return the required response payload."""
    state = graph.invoke(
        {
            "question": query, "context": [], "pages": [], "relevance": 0.0,
            "answer": "", "refused": False, "grounded": 0.0, "score": 0.0,
        }
    )
    return {
        "query": query,
        "final_answer": state["answer"],
        "retrieved_context_chunks": state["context"],
        "confidence_score": state["score"],
    }
