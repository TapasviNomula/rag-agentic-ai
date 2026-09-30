"""Runs the sample validation queries through the LangGraph pipeline."""
import json
import time

from src.graph import ask, build_rag_graph

QUERIES = [
    "What is the core definition of Agentic AI as outlined in the eBook?",
    "What are the main architectural components required to build agentic systems?",
    "What real-world industry use cases for Agentic AI are discussed in the eBook?",
    "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
    "What key challenges or limitations of Agentic AI are mentioned in the document?",
    "What is the capital of France?",  # out-of-scope: must refuse
]

if __name__ == "__main__":
    graph = build_rag_graph()
    for q in QUERIES:
        result = ask(graph, q)
        print("=" * 80)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        time.sleep(5)  # stay under free-tier requests-per-minute limits
