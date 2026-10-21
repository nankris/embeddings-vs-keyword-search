"""BM25 vs dense vs hybrid on BEIR SciFact (CPU). Metrics: nDCG@10, Recall@10.
Paired bootstrap CIs over queries. Resumable: embeddings cached to results/."""
import json, os, re, sys, time, math, hashlib
import numpy as np
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)

MODELS = {
    "dense_minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "dense_bge": "BAAI/bge-small-en-v1.5",
}
REVISIONS = {'sentence-transformers/all-MiniLM-L6-v2': '1110a243fdf4706b3f48f1d95db1a4f5529b4d41', 'BAAI/bge-small-en-v1.5': '5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'}
TOP_K = 10
CACHE_K = 100          # store top-100 ranks per query (metric can then use any k<=100)
SEED = 20261016
NBOOT = 10000

def load(name):
    with open(os.path.join(RES, name)) as f:
        return [json.loads(l) for l in f]

TOKEN_RE = re.compile(r"[a-z0-9]+")
def tok(s): return TOKEN_RE.findall(s.lower())

# ---------------- rankers ----------------
def run_bm25(corpus, queries):
    from rank_bm25 import BM25Okapi
    docs = [tok(c["title"] + " " + c["text"]) for c in corpus]
    bm = BM25Okapi(docs)
    out = {}
    for q in queries:
        scores = bm.get_scores(tok(q["text"]))
        idx = np.argsort(-scores)[:CACHE_K]
        out[q["id"]] = [corpus[i]["id"] for i in idx]
    return out

def embed_cached(key, texts, model_name, batch=64):
    cache = os.path.join(RES, f"emb_{key}_{hashlib.md5(model_name.encode()).hexdigest()[:8]}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    m = SentenceTransformer(model_name, revision=REVISIONS[model_name], device="cpu")
    E = m.encode(texts, batch_size=batch, normalize_embeddings=True,
                 show_progress_bar=False, convert_to_numpy=True)
    E = E.astype(np.float32)
    np.save(cache, E)
    print(f"  embedded {len(texts)} with {model_name} in {time.time()-t0:.1f}s -> {cache}")
    return E

def run_dense(key, corpus, queries):
    dc = [f'{c["title"]} {c["text"]}'.strip() for c in corpus]
    qc = [q["text"] for q in queries]
    DocE = embed_cached(key + "_docs", dc, MODELS[key])
    QE = embed_cached(key + "_queries", qc, MODELS[key])
    out = {}
    for i, q in enumerate(queries):
        s = DocE @ QE[i]                      # normalized -> cosine
        idx = np.argpartition(-s, CACHE_K)[:CACHE_K]
        idx = idx[np.argsort(-s[idx])]
        out[q["id"]] = [corpus[j]["id"] for j in idx]
    return out

def rrf(*runs, k=60, top=CACHE_K):
    """Reciprocal rank fusion of several ranked lists of doc ids."""
    agg = {}
    for run in runs:
        for qid, docs in run.items():
            acc = agg.setdefault(qid, {})
            for rank, d in enumerate(docs):
                acc[d] = acc.get(d, 0.0) + 1.0 / (k + rank + 1)
    return {qid: [d for d, _ in sorted(acc.items(), key=lambda x: -x[1])[:top]]
            for qid, acc in agg.items()}

# ---------------- metrics ----------------
def dcg(rels): return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))

def ndcg_at(ranked, rels, k=TOP_K):
    r = [rels.get(d, 0) for d in ranked[:k]]
    ideal = sorted(rels.values(), reverse=True)[:k]
    idc = dcg(ideal)
    return dcg(r) / idc if idc > 0 else 0.0

def recall_at(ranked, rels, k=TOP_K):
    n = sum(1 for v in rels.values() if v > 0)
    if n == 0: return 0.0
    hit = sum(1 for d in ranked[:k] if rels.get(d, 0) > 0)
    return hit / n

def score_run(run, qrels, metric):
    return {qid: metric(run.get(qid, []), rels) for qid, rels in qrels.items()}

# ---------------- paired bootstrap ----------------
def paired_boot(a, b, nboot=NBOOT, seed=SEED):
    """a,b: dict qid->score. Returns (mean_diff, lo, hi) for mean(a-b)."""
    ks = sorted(set(a) & set(b))
    A = np.array([a[k] for k in ks]); B = np.array([b[k] for k in ks])
    d = A - B
    rng = np.random.default_rng(seed)
    n = len(d)
    boot = np.empty(nboot)
    for i in range(nboot):
        s = rng.integers(0, n, n)
        boot[i] = d[s].mean()
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

# ---------------- paraphrase subset ----------------
def build_paraphrase_subset(corpus, queries, qrels, n=30):
    """Queries whose relevant abstract shares the fewest words with the query."""
    ctext = {c["id"]: c["title"] + " " + c["text"] for c in corpus}
    qtext = {q["id"]: q["text"] for q in queries}
    rels = {}
    for r in qrels:
        if r["score"] > 0: rels.setdefault(r["query_id"], []).append(r["corpus_id"])
    rows = []
    for qid, dids in rels.items():
        if qid not in qtext: continue
        qt = set(tok(qtext[qid]))
        if not qt: continue
        best = 0.0
        for d in dids:
            dt = set(tok(ctext.get(d, "")))
            if not dt: continue
            best = max(best, len(qt & dt) / len(qt | dt))   # max jaccard over gold docs
        rows.append((qid, best))
    rows.sort(key=lambda x: x[1])
    sub = [q for q, _ in rows[:n]]
    return sub, {q: j for q, j in rows[:n]}

# ---------------- main ----------------
def main():
    corpus, queries, qrels_l = load("corpus.jsonl"), load("queries.jsonl"), load("qrels.jsonl")
    qrels = {}
    for r in qrels_l:
        if r["score"] > 0: qrels.setdefault(r["query_id"], {})[r["corpus_id"]] = r["score"]
    print(f"corpus={len(corpus)} queries_with_qrels={len(qrels)}")

    runs = {}
    for key in ["bm25", "dense_minilm", "dense_bge"]:
        path = os.path.join(RES, f"run_{key}.json")
        if os.path.exists(path):
            with open(path) as f: runs[key] = json.load(f)
            print("resumed", key, flush=True)
        else:
            print("running", key, flush=True)
            runs[key] = run_bm25(corpus, queries) if key == "bm25" else run_dense(key, corpus, queries)
            with open(path + ".tmp", "w") as f: json.dump(runs[key], f)
            os.replace(path + ".tmp", path)
            print("saved", key, flush=True)
    print("[4/4] hybrid RRF(bm25, dense_minilm) ...")
    runs["hybrid"] = rrf(runs["bm25"], runs["dense_minilm"])

    for k, v in runs.items():
        with open(os.path.join(RES, f"run_{k}.json"), "w") as f: json.dump(v, f)
    print("saved runs")

    sub, jac = build_paraphrase_subset(corpus, queries, qrels_l, 30)
    qtext = {q["id"]: q["text"] for q in queries}
    with open(os.path.join(RES, "paraphrase_subset.json"), "w") as f:
        json.dump({"query_ids": sub, "jaccard": jac,
                   "texts": {q: qtext[q] for q in sub}}, f, indent=1)
    print(f"paraphrase subset: {len(sub)} queries, jaccard<= {max(jac.values()):.3f}")

    # all queries + subset
    out_rows = []
    CIs = {}
    for subname, qids in [("all", sorted(qrels)), ("paraphrase", sub)]:
        qr = {q: qrels[q] for q in qids if q in qrels}
        sys_scores = {s: {} for s in runs}
        for s, run in runs.items():
            sys_scores[s]["nDCG@10"] = score_run(run, qr, ndcg_at)
            sys_scores[s]["Recall@10"] = score_run(run, qr, recall_at)
        for s in runs:
            out_rows.append({"subset": subname, "system": s, "n": len(qr),
                             "ndcg@10": float(np.mean(list(sys_scores[s]["nDCG@10"].values()))),
                             "recall@10": float(np.mean(list(sys_scores[s]["Recall@10"].values())))})
        if subname == "all":
            for metric in ["nDCG@10", "Recall@10"]:
                for a, b in [("dense_bge", "bm25"), ("dense_minilm", "bm25"),
                             ("hybrid", "bm25"), ("hybrid", "dense_minilm"),
                             ("hybrid", "dense_bge"), ("dense_bge", "dense_minilm")]:
                    m, lo, hi = paired_boot(sys_scores[a][metric], sys_scores[b][metric])
                    CIs.setdefault(metric, {})[f"{a}-{b}"] = {"diff": m, "lo": lo, "hi": hi}
        else:
            CIs.setdefault("paraphrase", {})
            for metric in ["nDCG@10", "Recall@10"]:
                for a, b in [("dense_bge", "bm25"), ("dense_minilm", "bm25"), ("hybrid", "bm25")]:
                    m, lo, hi = paired_boot(sys_scores[a][metric], sys_scores[b][metric])
                    CIs["paraphrase"][f"{metric}|{a}-{b}"] = {"diff": m, "lo": lo, "hi": hi}

    with open(os.path.join(RES, "metrics.json"), "w") as f:
        json.dump({"rows": out_rows, "cis": CIs}, f, indent=1)

    import csv
    with open(os.path.join(RES, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subset", "system", "n", "ndcg@10", "recall@10"], lineterminator="\n")
        w.writeheader(); w.writerows(out_rows)

    print("\n=== RESULTS (all queries, n=%d) ===" % len(qrels))
    for r in out_rows:
        if r["subset"] == "all":
            print(f'{r["system"]:14s} nDCG@10={r["ndcg@10"]:.4f}  Recall@10={r["recall@10"]:.4f}')
    print("\n=== PAIRED BOOTSTRAP 95% CI (all queries) ===")
    for metric, d in CIs.items():
        if metric == "paraphrase": continue
        for pair, v in sorted(d.items()):
            print(f'{metric:10s} {pair:24s} diff={v["diff"]:+.4f} [{v["lo"]:+.4f},{v["hi"]:+.4f}]')
    print("\n=== paraphrase subset (30 low-overlap queries) ===")
    for r in out_rows:
        if r["subset"] == "paraphrase":
            print(f'{r["system"]:14s} nDCG@10={r["ndcg@10"]:.4f}  Recall@10={r["recall@10"]:.4f}')
    for pair, v in sorted(CIs.get("paraphrase", {}).items()):
        print(f'{pair:34s} diff={v["diff"]:+.4f} [{v["lo"]:+.4f},{v["hi"]:+.4f}]')

if __name__ == "__main__":
    main()
