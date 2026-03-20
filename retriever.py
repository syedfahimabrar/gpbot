"""
BM25-based FAQ retrieval.
Given a query and optionally an intent, finds the best matching FAQ answer.
"""

import json
from rank_bm25 import BM25Okapi
from model import tokenize


def load_faq_kb(path="data/faq_kb.json"):
    with open(path) as f:
        return json.load(f)


def search_faq(query, knowledge_base, intent=None):
    """
        Search FAQ knowledge base using BM25. Optionally filter by intent.

        Args:
            query: User query string
            knowledge_base: List of FAQs
            intent: Optional intent filter

        Returns:
            Best matching answer string
    """
    if intent:
        filtered = [faq for faq in knowledge_base if faq["intent"] == intent]
        if not filtered:
            filtered = knowledge_base
    else:
        filtered = knowledge_base

    tokenized_docs = [tokenize(faq["question"]) for faq in filtered]
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenize(query))
    best_idx = scores.argmax()

    return filtered[best_idx]["answer"]


# test
if __name__ == "__main__":
    kb = load_faq_kb()
    queries = [
        ("How to check balance?", "balance_check"),
        ("internet package price", "data_package"),
        ("sim hariye geche", "sim_replace"),
    ]
    for query, intent in queries:
        answer = search_faq(query, kb, intent)
        print(f"Q: {query}\nA: {answer[:100]}...\n")
