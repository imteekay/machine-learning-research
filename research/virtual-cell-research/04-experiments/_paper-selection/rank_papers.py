"""Rank each roadmap phase's readings by citation count (Semantic Scholar).

Usage: python -I rank_papers.py <roadmap.md> <out.json>

Only papers in METHOD_KEYWORDS-matched titles count as "method" papers; the
curated NON_METHOD list removes reviews, tools, atlases and datasets.
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request

S2 = "https://api.semanticscholar.org/graph/v1/paper/{}?fields=title,citationCount,year"

# Titles (substring, lowercase) excluded from the "method paper" ranking:
# reviews, perspectives, libraries/tools, atlases, datasets, benchmarks, critiques.
NON_METHOD = [
    "how to build the virtual cell", "virtual cells: predict", "towards multimodal foundation",
    "harnessing the deep learning", "transformers in single-cell omics", "interpretation, extrapolation",
    "flow matching for generative", "mini-review", "scanpy", "scvi-tools", "pertpy", "shortcomings of silhouette",
    "batch effects remain", "scdataset", "rapids-singlecell", "metadata as language", "cellxgene", "tabula sapiens",
    "virtual cell atlas", "scbasecount", "heca", "human cell atlas", "cellnexus", "private information leakage",
    "zero-shot evaluation", "parameter-free", "fundamental limitations", "scaling up training", "evaluating the role",
    "scaling recipes", "metric mirages", "limitations of cell embedding", "delineating", "systematic evaluation",
    "context-dependent utility", "vcbench", "cellverse", "scmbench", "sctranslation", "multitask benchmarking",
    "norman", "exploring genetic interaction", "mapping information-rich", "x-atlas", "tahoe-100m", "cytokine dictionary",
    "perturbase", "does not yet outperform", "one pca still", "systema", "do outperform uninformative", "perturbench",
    "pertevalscfm", "perteval", "benchmarking", "towards principled", "far from straightforward", "scarchon",
    "virtual cell challenge", "evaluating the ability", "reusability report", "a perspective on", "toward next-generation",
    "multimodal spatial omics", "spatial omics at the forefront", "what makes a virtual cell", "world models for biomedicine",
    "a world model of the virtual cell", "virtual cells need context", "harnessing ai", "the virtual cell", "grow ai virtual",
    "fifteen challenges", "predicting spatial transcriptomics from histology", "stp-bench", "empirical evaluation",
    "how to build an ai-driven", "ai virtual cells for drug", "ai-driven virtual cell", "from equations to agents",
    "next-generation virtual cell", "perturbation cell and tissue atlas", "agentic systems are adept",
    "spatial crispr genomics", "transcription factor networks disproportionately", "foundation models improve perturbation",
]


def phases(md: str):
    parts = re.split(r"^## Phase (\d): ", md, flags=re.M)
    for i in range(1, len(parts), 2):
        body = parts[i + 1].split("### Hands-on Milestone")[0]
        yield int(parts[i]), body


def paper_id(url: str):
    if m := re.search(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})", url):
        return f"arXiv:{m.group(1)}"
    if m := re.search(r"doi\.org/(10\.[^\s)]+)", url):
        return f"DOI:{m.group(1)}"
    if m := re.search(r"nature\.com/articles/([\w.-]+)", url):
        return f"DOI:10.1038/{m.group(1)}"
    if m := re.search(r"biorxiv\.org/content/(10\.\d+/[\d.]+)", url):
        return f"DOI:{m.group(1)}"
    if m := re.search(r"science\.org/doi/(10\.[^\s)]+)", url):
        return f"DOI:{m.group(1)}"
    if m := re.search(r"(?:pnas|embopress)\.org/doi/(?:full/)?(10\.[^\s)]+)", url):
        return f"DOI:{m.group(1)}"
    if m := re.search(r"genomebiology\.biomedcentral\.com/articles/(10\.[^\s)]+)", url):
        return f"DOI:{m.group(1)}"
    return f"URL:{url}"


def fetch(pid: str):
    url = S2.format(urllib.parse.quote(pid, safe=":/"))
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "paper-ranker"}), timeout=20) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            return None
        except Exception:
            return None
    return None


def main(roadmap, out):
    md = open(roadmap).read()
    result = {}
    for ph, body in phases(md):
        rows = []
        for title, url in re.findall(r"- \[ \] \[([^\]]+)\]\((https?://[^)]+)\)", body):
            low = title.lower()
            method = not any(k in low for k in NON_METHOD)
            info = fetch(paper_id(url))
            time.sleep(1.1)
            rows.append({"title": title, "url": url, "method": method,
                         "citations": (info or {}).get("citationCount"), "year": (info or {}).get("year")})
            print(ph, rows[-1]["citations"], method, title[:70], file=sys.stderr)
        rows.sort(key=lambda r: -(r["citations"] or -1))
        result[ph] = rows
    json.dump({"queried": time.strftime("%Y-%m-%d"), "phases": result}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:3])
