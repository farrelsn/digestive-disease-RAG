"""
Step 4: Test retrieval

For every question in test_questions.json, checks whether the search returns a chunk
from one of the expected pages in its top results. Runs semantic, BM25 and hybrid
search so you can compare them.

Scores:
  Hit rate: how many questions got a right page in the top TOP_K results (higher is better)
  MRR:      how HIGH the first right page is ranked, on average
            1.0 = always #1, 0.5 = usually #2, 0.33 = usually #3

Run: python evaluate.py
"""
import json

from chunking import load_chunks
from retrieve import TOP_K, build_hybrid_retriever

TEST_FILE = "data/test_questions.json"


def first_correct_rank(chunks, expected_documents):
    """Position (1, 2, 3...) of the first chunk from an expected page, or None if there is none."""
    for rank, chunk in enumerate(chunks, start=1):
        if chunk.metadata["document_id"] in expected_documents:
            return rank
    return None


def evaluate(name, search, tests):
    hits = 0
    reciprocal_rank_total = 0
    misses = []

    for test in tests:
        chunks = search(test["question"])[:TOP_K]
        rank = first_correct_rank(chunks, test["expected_documents"])
        if rank:
            hits += 1
            reciprocal_rank_total += 1 / rank
        else:
            misses.append((test, chunks))

    print(f"\n{name}: hit rate {hits}/{len(tests)} ({hits / len(tests):.0%}) | MRR {reciprocal_rank_total / len(tests):.2f}")
    for test, chunks in misses:
        found_pages = dict.fromkeys(chunk.metadata["document_id"] for chunk in chunks)
        print(f"  MISSED: {test['question']}")
        print(f"    expected: {', '.join(test['expected_documents'])}")
        print(f"    got:      {', '.join(found_pages)}")


if __name__ == "__main__":
    with open(TEST_FILE, encoding="utf-8") as file:
        tests = json.load(file)

    # Catch typos in the test file: every expected page must exist in the chunks.
    known_documents = {chunk.metadata["document_id"] for chunk in load_chunks()}
    for test in tests:
        for document_id in test["expected_documents"]:
            if document_id not in known_documents:
                raise ValueError(f"Unknown page '{document_id}' in question: {test['question']}")

    hybrid = build_hybrid_retriever()
    semantic_search, keyword_search = hybrid.retrievers

    evaluate("Semantic only", semantic_search.invoke, tests)
    evaluate("Keyword only (BM25)", keyword_search.invoke, tests)
    evaluate("Hybrid", hybrid.invoke, tests)
