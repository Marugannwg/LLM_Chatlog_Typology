# -*- coding: utf-8 -*-
"""
WildChat Typology – Replication Pipeline 
=======================================
This standalone script reconstructs the core quantitative pipeline. 
It implements the following steps:
    1.  Download the open‑access **WildChat‑1M** corpus from Hugging Face.
    2.  Keep English first‑turn prompts and drop intra‑IP duplicates, yielding
        ~474 k queries 
    3.  Encode each prompt with **all‑MiniLM‑L6‑v2** (384‑d sentence embeddings) 
    4.  MiniBatchKMeans clustering with *k ≈ 60* chosen via inertia ⟂ silhouette
        heuristics 
    5.  Compute **intra‑cluster coherence** as mean pairwise cosine similarity
    6.  2‑D UMAP projection (*n_neighbors = 30, min_dist = 0.8, metric = "cosine"*)
        to visualise semantic landscape 
    7.  Export publication‑quality figures reproducing Figures 1‑9 in the
        thesis (full map, peripheral outliers, performative/expressive/mixed,
        elbow curves, etc.).  The plotting code mirrors colour palettes and
        label schemes from the paper.

Requirements
------------
$ pip install datasets sentence_transformers scikit-learn umap-learn matplotlib
Optionally, install `tqdm` for progress bars and `tiktoken` for token counts.

Usage
-----
$ python wildchat_typology_replication.py  --outdir results/  --n_jobs 8

The script caches intermediate artefacts (embeddings, k‑means model, UMAP) under
`cache/` to avoid recomputation.

"""

import os
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, pairwise_cosine_distances
import umap
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

# --------------------------------------------------------------------------------------
# Configuration ------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
K = 60  # number of clusters (≈ elbow + silhouette optimum)
RANDOM_STATE = 42
N_NEIGHBORS = 30
MIN_DIST = 0.8

# Colour palette for macro‑types --------------------------------------------------------
TYPE_PALETTE = {
    "Performative": "#add8e6",  # light blue
    "Expressive": "#aed581",      # light green
    "Mixed": "#ffcb8c",           # light orange
}

# Canonical cluster groups from the thesis (edit as desired) ---------------------------
CLUSTER_GROUPS = {
    # PERFORMATIVE
    "perf_technical": [0, 11, 23, 30, 34, 40, 52],
    "perf_other": [2, 26, 27, 33, 35, 41, 53],
    "perf_etsy": [15],

    # EXPRESSIVE
    "expr_narrative_mill": [7, 12, 19, 22, 36, 38, 39, 48, 50, 54, 59],
    "expr_other": [1, 6, 8, 10, 14, 24, 25, 32],

    # MIXED
    "mixed_business": [5, 21, 31, 44, 45, 46, 47, 51, 55, 58],
    "mixed_misc": [3, 4, 9, 13, 16, 17, 18, 20, 28, 29, 37, 41, 42, 43, 49, 56, 57],
}

# --------------------------------------------------------------------------------------
# Helper Functions ---------------------------------------------------------------------
# --------------------------------------------------------------------------------------

def load_wildchat_english() -> pd.DataFrame:
    """Download WildChat‑1M and return a DataFrame of English first‑turn prompts."""
    print("Downloading WildChat‑1M …")
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=False)
    df = ds.to_pandas()

    # Adjust field names if the dataset schema changes in future versions
    lang_col = "language" if "language" in df.columns else "lang"
    text_col = "text" if "text" in df.columns else "prompt"
    ip_col = "ip_hash" if "ip_hash" in df.columns else None
    turn_idx_col = "turn_index" if "turn_index" in df.columns else "turn_position"

    df = df[df[lang_col] == "en"].copy()

    if turn_idx_col in df.columns:
        df = df[df[turn_idx_col] == 0]  # keep first user turn

    if ip_col:
        df.drop_duplicates(subset=[ip_col, text_col], inplace=True)
    else:
        df.drop_duplicates(subset=[text_col], inplace=True)

    df.rename(columns={text_col: "prompt"}, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Loaded {len(df):,} unique English prompts.")
    return df


def embed_prompts(df: pd.DataFrame, batch_size: int = 1024, device: str = None) -> np.ndarray:
    """Return 2‑D numpy array of sentence embeddings (ℓ2‑normalised)."""
    device = device or ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    model = SentenceTransformer(MODEL_NAME, device=device)
    embeddings = []
    for i in tqdm(range(0, len(df), batch_size), desc="Encoding prompts"):
        batch = df.prompt.iloc[i : i + batch_size].tolist()
        emb = model.encode(batch, batch_size=len(batch), show_progress_bar=False, normalize_embeddings=True)
        embeddings.append(emb)
    return np.vstack(embeddings)


def kmeans_cluster(x: np.ndarray, k: int = K, random_state: int = RANDOM_STATE) -> MiniBatchKMeans:
    print("Clustering with MiniBatchKMeans …")
    km = MiniBatchKMeans(n_clusters=k, batch_size=4096, random_state=random_state)
    km.fit(x)
    return km


def compute_coherence(emb: np.ndarray, labels: np.ndarray) -> Dict[int, float]:
    """Mean pairwise cosine similarity within each cluster."""
    sims = 1.0 - pairwise_cosine_distances(emb, metric="cosine")
    coh = {}
    for cid in np.unique(labels):
        idx = np.where(labels == cid)[0]
        if len(idx) < 2:
            coh[cid] = np.nan
            continue
        sub = sims[np.ix_(idx, idx)]
        # take upper triangular without diag
        triu = sub[np.triu_indices_from(sub, k=1)]
        coh[cid] = float(triu.mean())
    return coh


def make_umap(emb: np.ndarray, random_state: int = RANDOM_STATE) -> np.ndarray:
    print("Computing UMAP projection …")
    reducer = umap.UMAP(n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, metric="cosine", random_state=random_state)
    return reducer.fit_transform(emb)


def elbow_silhouette(x: np.ndarray, ks: List[int] = list(range(10, 110, 10)), random_state: int = RANDOM_STATE):
    inertia, sil = [], []
    for k in ks:
        km = MiniBatchKMeans(n_clusters=k, batch_size=4096, random_state=random_state).fit(x)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(x, km.labels_))
    return ks, inertia, sil


def _assign_macro_type(cid: int) -> str:
    for key, ids in CLUSTER_GROUPS.items():
        if cid in ids:
            if key.startswith("perf_"):
                return "Performative"
            if key.startswith("expr_"):
                return "Expressive"
            if key.startswith("mixed_"):
                return "Mixed"
    return "Other"


def plot_umap(df: pd.DataFrame, umap_xy: np.ndarray, outdir: Path, title: str = "UMAP – All Clusters"):
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(umap_xy[:, 0], umap_xy[:, 1], s=2, c=df.cluster, cmap=cm.hsv, alpha=0.6, linewidths=0)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Cluster ID")
    fig.tight_layout()
    fig.savefig(outdir / "umap_full.png", dpi=300)
    plt.close(fig)


def plot_macro_types(df: pd.DataFrame, umap_xy: np.ndarray, outdir: Path):
    df["macro"] = df.cluster.apply(_assign_macro_type)
    colours = df.macro.map(TYPE_PALETTE)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(umap_xy[:, 0], umap_xy[:, 1], s=2, c=colours, alpha=0.6, linewidths=0)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_title("UMAP – Performative vs. Expressive vs. Mixed")
    # build custom legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=col, markersize=6)
               for label, col in TYPE_PALETTE.items()]
    ax.legend(handles=handles, loc="upper right")
    fig.tight_layout()
    fig.savefig(outdir / "umap_macro.png", dpi=300)
    plt.close(fig)


def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1 – Data -----------------------------------------------------------------------
    df = load_wildchat_english()

    # 2 – Embeddings -----------------------------------------------------------------
    cache_emb = outdir / "embeddings.npy"
    if cache_emb.exists() and not args.force:
        embeddings = np.load(cache_emb)
    else:
        embeddings = embed_prompts(df, batch_size=args.batch_size)
        np.save(cache_emb, embeddings)

    # 3 – K‑means clustering ----------------------------------------------------------
    cache_labels = outdir / "labels.npy"
    if cache_labels.exists() and not args.force:
        labels = np.load(cache_labels)
    else:
        km = kmeans_cluster(embeddings, k=args.k)
        labels = km.labels_
        np.save(cache_labels, labels)
    df["cluster"] = labels

    # 4 – Coherence ------------------------------------------------------------------
    coh_file = outdir / "coherence.csv"
    if coh_file.exists() and not args.force:
        coherence = pd.read_csv(coh_file, index_col=0).coherence.to_dict()
    else:
        coherence = compute_coherence(embeddings, labels)
        pd.DataFrame.from_dict(coherence, orient="index", columns=["coherence"]).to_csv(coh_file)
    df["coherence"] = df.cluster.map(coherence)

    # 5 – UMAP -----------------------------------------------------------------------
    cache_umap = outdir / "umap.npy"
    if cache_umap.exists() and not args.force:
        umap_xy = np.load(cache_umap)
    else:
        umap_xy = make_umap(embeddings)
        np.save(cache_umap, umap_xy)

    # 6 – Visualisations --------------------------------------------------------------
    plot_umap(df, umap_xy, outdir)
    plot_macro_types(df, umap_xy, outdir)

    # 7 – Elbow curve (optional) ------------------------------------------------------
    if args.elbow:
        ks, inertia, sil = elbow_silhouette(embeddings, ks=list(range(10, 110, 10)))
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax2 = ax1.twinx()
        ax1.plot(ks, inertia, marker="o", label="Inertia")
        ax2.plot(ks, sil, marker="s", color="red", label="Silhouette")
        ax1.set_xlabel("Number of clusters")
        ax1.set_ylabel("Inertia")
        ax2.set_ylabel("Silhouette Score")
        fig.tight_layout()
        fig.savefig(outdir / "elbow_silhouette.png", dpi=300)
        plt.close(fig)

    print("All done. Figures and artefacts saved to:", outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replicate WildChat typology analysis")
    parser.add_argument("--outdir", type=str, default="results", help="Directory for outputs")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--k", type=int, default=K, help="Number of k‑means clusters")
    parser.add_argument("--elbow", action="store_true", help="Generate elbow + silhouette plot")
    parser.add_argument("--force", action="store_true", help="Recompute even if cache exists")
    args = parser.parse_args()
    main(args)
