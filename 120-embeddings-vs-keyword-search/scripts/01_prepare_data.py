"""Download BEIR SciFact and freeze it to results/ as jsonl (resumable)."""
import json, os
from datasets import load_dataset

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

def write(name, rows):
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote", p, len(rows))

def main():
    corpus_p = os.path.join(OUT, "corpus.jsonl")
    qrels_p = os.path.join(OUT, "qrels.jsonl")
    if all(os.path.exists(os.path.join(OUT, n)) for n in ["corpus.jsonl", "queries.jsonl", "qrels.jsonl"]):
        print("already prepared"); return
    corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries = load_dataset("BeIR/scifact", "queries", split="queries")
    qrels = load_dataset("BeIR/scifact-qrels", split="test")
    write("corpus.jsonl", [{"id": str(r["_id"]), "title": r["title"] or "", "text": r["text"]} for r in corpus])
    write("queries.jsonl", [{"id": str(r["_id"]), "text": r["text"]} for r in queries])
    write("qrels.jsonl", [{"query_id": str(r["query-id"]), "corpus_id": str(r["corpus-id"]), "score": int(r["score"])} for r in qrels])

if __name__ == "__main__":
    main()
