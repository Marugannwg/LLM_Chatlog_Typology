# WildChat Typology Replication

This repository contains a reproduction of the quantitative pipeline described in Tianfang Zhu’s 2025 thesis on emergent user–LLM interaction types.  It walks from raw *WildChat‑1M* logs to the final cluster tables and UMAP figures that appear in the paper.

---

## Folder & file map

| Stage                             | File(s)                                             | Purpose                                                                                                                                                                                                                                                                                    |
| --------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **1 · Embedding**                 | `full_embedding.ipynb`                              | Downloads the corpus from 🤗 *Hugging Face*, filters English first‑turn prompts, and embeds them with **all‑MiniLM‑L6‑v2**.  Produces two artefacts:<br>  • `wildchat_subset.parquet` – tidy DataFrame with metadata<br>  • `wildchat_embeddings.jsonl` – 384‑d float32 matrix (≈ 1 GB for full set) |
| **2 · Pipeline overview**         | `pipeline_overview.py`                              | Lightweight CLI wrapper (the one generated in‑house) that chains the whole process end‑to‑end.  Use it when you want a single command to rebuild everything.                                                                                                                               |
| **3 · Exploratory data analysis** | `exploration_example.ipynb`, `table_plotting.ipynb` | Show‑and‑tell notebooks that replicate descriptive tables and sanity‑check plots (see § 4.1 of the thesis).                                                                                                                                                                                |
| **4 · Spatial visualisation**     | `umap_plotting.ipynb`                               | Projects the 384‑d space to 2‑d UMAP using the thesis hyper‑parameters (n\_neighbors = 40, min\_dist = 0.02, metric = "cosine").  Generates Figures 1–7 and writes them to `figures/`.                                                                                                     |
| *(helper)*                        | `umap_subset.py`                                    | Convenience script to slice the full embedding/metadata set and dump a smaller `.npy + .csv` pair for quick experimentation.                                                                                                                                                               |

---

## Quick‑start

```bash
# 0.  Clone and create an isolated env (optional)
conda create -n wildchat python=3.10 && conda activate wildchat

# 1.  Install requirements
pip install -r requirements.txt  # see below for the list

# 2.  Run the full pipeline in one go (GPU strongly recommended)
python pipeline_overview.py --full   # ~45 min on an RTX 4090

# 3.  Or step through the notebooks if you prefer an interactive walk‑through
jupyter lab full_embedding.ipynb
```

### Requirements

* Python ≥ 3.9
* pandas, numpy, scikit‑learn, sentence‑transformers, umap‑learn, matplotlib, seaborn, tqdm
* (optional) cuBLAS/cuDNN‑enabled PyTorch for GPU‑accelerated embedding & UMAP

---

## Licence

Code is MIT‑licensed; the **WildChat‑1M** dataset is released by AllenAI under its own terms—see the dataset card on Hugging Face.

---

*Questions or issues?  Open a GitHub discussion.*
