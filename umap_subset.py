import pandas as pd
import numpy as np
import umap
import json
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch


"""
Helper functions for UMAP plotting
"""
def fit_umap_on_clusters(
    raw_df: pd.DataFrame,
    embeddings: List[dict],
    cluster_col: str = 'cluster',
    clusters: Optional[List[int]] = None,
    n_neighbors: int = 30,
    min_dist: float = 0.8,
    n_components: int = 3,
    metric: str = 'cosine',
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit UMAP on a subset of data defined by clusters.
    If clusters is None, fit on all data.
    Returns: DataFrame with UMAP coordinates for the selected subset (all original columns + umap_x/y/z).
    """
    # Deduplicate embeddings by conversation_hash
    conversation_hash_to_index = {}
    unique_embeddings = []
    for item in embeddings:
        conv_hash = item['conversation_hash']
        if conv_hash not in conversation_hash_to_index:
            conversation_hash_to_index[conv_hash] = len(unique_embeddings)
            unique_embeddings.append(item)
    embeddings = unique_embeddings

    # Extract embedding vectors and hashes
    embedding_vectors = np.array([item['embedding'] for item in embeddings])
    embedding_hashes = [item['conversation_hash'] for item in embeddings]

    # Map conversation_hash to index in raw_df
    raw_df_hash_dict = dict(zip(raw_df['conversation_hash'], range(len(raw_df))))

    # Find intersection of hashes
    common_hashes = []
    common_indices = []
    for i, conv_hash in enumerate(embedding_hashes):
        if conv_hash in raw_df_hash_dict:
            common_hashes.append(conv_hash)
            common_indices.append(i)

    filtered_vectors = embedding_vectors[common_indices]
    filtered_raw_df = raw_df[raw_df['conversation_hash'].isin(common_hashes)].copy()

    # Optionally filter by clusters
    if clusters is not None:
        filtered_raw_df = filtered_raw_df[filtered_raw_df[cluster_col].isin(clusters)].copy()
        # Find indices in filtered_vectors that match filtered_raw_df
        filtered_hashes = set(filtered_raw_df['conversation_hash'])
        filtered_indices = [i for i, h in enumerate(common_hashes) if h in filtered_hashes]
        filtered_vectors = filtered_vectors[filtered_indices]
        filtered_raw_df = filtered_raw_df.reset_index(drop=True)
    else:
        filtered_raw_df = filtered_raw_df.reset_index(drop=True)

    # UMAP fitting (landmark + transform for speed, as in notebook)
    rng = np.random.default_rng(random_state)
    landmark_frac = 0.05
    n_landmarks = max(1, int(len(filtered_vectors) * landmark_frac))
    landmark_idx = rng.choice(len(filtered_vectors), n_landmarks, replace=False)
    rest_idx = np.setdiff1d(np.arange(len(filtered_vectors)), landmark_idx)
    X_landmark = filtered_vectors[landmark_idx]
    X_rest = filtered_vectors[rest_idx]

    mapper = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        init='random',
        random_state=random_state,
        n_jobs=-1,
    )
    Z_landmark = mapper.fit_transform(X_landmark)
    Z_rest = mapper.transform(X_rest)

    all_coords = np.empty((len(filtered_vectors), n_components), dtype=np.float32)
    all_coords[landmark_idx] = Z_landmark
    all_coords[rest_idx] = Z_rest

    filtered_raw_df['umap_x'] = all_coords[:, 0]
    filtered_raw_df['umap_y'] = all_coords[:, 1]
    if n_components > 2:
        filtered_raw_df['umap_z'] = all_coords[:, 2]
    return filtered_raw_df

# Example usage (uncomment and adapt as needed):
# raw_df = pd.read_csv('first_query_clustered.csv')
# with open('wildchat_embeddings.jsonl', 'r') as f:
#     embeddings = [json.loads(line) for line in f]
# df_umap = fit_umap_on_clusters(raw_df, embeddings, clusters=[1,2,3]) 

def get_distinct_colors(n=60):
    """
    Generate n visually distinct colors, avoiding grey and white.
    Uses a combination of tab20, tab20b, tab20c, Set1, Set2, Set3, and hsv colormaps.
    """
    # Combine several colormaps, skipping light/grey colors
    cmaps = [mpl.cm.get_cmap(name) for name in ['tab20', 'tab20b', 'tab20c', 'Set1', 'Set2', 'Set3', 'hsv']]
    colors = []
    for cmap in cmaps:
        for i in range(cmap.N):
            color = cmap(i / (cmap.N - 1))
            # Exclude colors that are too light (close to white) or too grey
            if not (sum(color[:3]) > 2.7 or (abs(color[0]-color[1]) < 0.08 and abs(color[1]-color[2]) < 0.08)):
                colors.append(color)
            if len(colors) >= n:
                break
        if len(colors) >= n:
            break
    # If not enough, fill with hsv
    while len(colors) < n:
        color = mpl.cm.hsv(len(colors) / n)
        colors.append(color)
    return colors[:n]

def plot_selected_clusters(
    df,
    highlight_ids,
    cluster_col='cluster',
    annotate=True,
    figsize=(12, 10),
    pt_size=12,
    alpha_bg=0.08,
    alpha_fg=0.85,
    save_as=None,
    title='UMAP Projection – Selected Clusters Highlighted',
    legend_title='Highlighted clusters',
    fixed_palette=False,
):
    """
    df            : DataFrame with 'umap_x', 'umap_y', cluster_col
    highlight_ids : iterable of cluster IDs to show in colour
    fixed_palette : if True, clusters 0-59 always get the same color from get_distinct_colors
    """
    highlight_ids = list(sorted(set(highlight_ids)))
    if fixed_palette:
        color_list = get_distinct_colors(60)
        lut = {cid: color_list[cid % 60] for cid in highlight_ids}
    else:
        color_list = get_distinct_colors(max(60, len(highlight_ids)))
        lut = {cid: color_list[i % len(color_list)] for i, cid in enumerate(highlight_ids)}

    fig, ax = plt.subplots(figsize=figsize)

    # Background: all points in grey
    ax.scatter(df['umap_x'], df['umap_y'],
               color='lightgrey', s=pt_size,
               alpha=alpha_bg, linewidths=0)

    # Foreground: plot each highlighted cluster individually
    legend_handles = []
    for i, cid in enumerate(highlight_ids):
        mask = df[cluster_col] == cid
        colour = lut[cid]
        ax.scatter(df.loc[mask, 'umap_x'],
                   df.loc[mask, 'umap_y'],
                   color=colour, s=pt_size,
                   alpha=alpha_fg, label=f'Cluster {cid}',
                   linewidths=0)
        if annotate:
            cx = df.loc[mask, 'umap_x'].mean()
            cy = df.loc[mask, 'umap_y'].mean()
            ax.text(cx, cy, str(cid),
                    ha='center', va='center',
                    fontsize=10, weight='bold',
                    color='black', alpha=0.9,
                    bbox=dict(boxstyle='round,pad=0.25',
                              fc='white', ec='none', alpha=0.6))
        legend_handles.append(Patch(facecolor=colour, label=f'{cid}'))

    # Legend
    ax.legend(handles=legend_handles, title=legend_title,
              framealpha=0.9, fontsize=14, title_fontsize=16,
              loc='upper right')

    # Axis & title styling
    ax.set_xlabel('UMAP Dim 1', fontsize=16)
    ax.set_ylabel('UMAP Dim 2', fontsize=16)
    ax.set_title(title, fontsize=24, pad=15, fontweight='bold')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

    # Count + percentage text
    sel_mask = df[cluster_col].isin(highlight_ids)
    n_sel = int(sel_mask.sum())
    pct_sel = n_sel / len(df) * 100
    ax.text(0.02, 0.025,
            f'n = {n_sel:,}  ({pct_sel:0.1f} %)',
            transform=ax.transAxes,
            ha='left', va='bottom',
            fontsize=16, weight='bold',
            bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.3))

    plt.tight_layout()
    if save_as:
        fig.savefig(save_as, dpi=300, bbox_inches='tight')
    plt.show()

def plot_clusters_with_custom_colors(
    df,
    highlight_ids,
    color_scheme,
    cluster_col='cluster',
    annotate=True,
    figsize=(12, 10),
    pt_size=12,
    alpha_bg=0.08,
    alpha_fg=0.85,
    save_as=None,
    title='UMAP Projection – Custom Cluster Colors',
    legend_title='Highlighted clusters',
):
    """
    df            : DataFrame with 'umap_x', 'umap_y', cluster_col
    highlight_ids : iterable of cluster IDs to show in colour
    color_scheme  : dict mapping cluster_id to color (name or RGB tuple)
    """
    highlight_ids = list(sorted(set(highlight_ids)))
    # Use the provided color scheme, fallback to grey
    lut = {cid: color_scheme.get(cid, 'grey') for cid in highlight_ids}

    fig, ax = plt.subplots(figsize=figsize)

    # Background: all points in lightgrey
    ax.scatter(df['umap_x'], df['umap_y'],
               color='lightgrey', s=pt_size,
               alpha=alpha_bg, linewidths=0)

    # Foreground: plot each highlighted cluster individually
    legend_handles = []
    for cid in highlight_ids:
        mask = df[cluster_col] == cid
        colour = lut[cid]
        ax.scatter(df.loc[mask, 'umap_x'],
                   df.loc[mask, 'umap_y'],
                   color=colour, s=pt_size,
                   alpha=alpha_fg, label=f'Cluster {cid}',
                   linewidths=0)
        if annotate:
            cx = df.loc[mask, 'umap_x'].mean()
            cy = df.loc[mask, 'umap_y'].mean()
            ax.text(cx, cy, str(cid),
                    ha='center', va='center',
                    fontsize=10, weight='bold',
                    color='black', alpha=0.9,
                    bbox=dict(boxstyle='round,pad=0.25',
                              fc='white', ec='none', alpha=0.6))
        legend_handles.append(Patch(facecolor=colour, label=f'{cid}'))

    # Legend
    ax.legend(handles=legend_handles, title=legend_title,
              framealpha=0.9, fontsize=14, title_fontsize=16,
              loc='upper right')

    # Axis & title styling
    ax.set_xlabel('UMAP Dim 1', fontsize=16)
    ax.set_ylabel('UMAP Dim 2', fontsize=16)
    ax.set_title(title, fontsize=24, pad=15, fontweight='bold')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

    # Count + percentage text
    sel_mask = df[cluster_col].isin(highlight_ids)
    n_sel = int(sel_mask.sum())
    pct_sel = n_sel / len(df) * 100
    ax.text(0.02, 0.025,
            f'n = {n_sel:,}  ({pct_sel:0.1f} %)',
            transform=ax.transAxes,
            ha='left', va='bottom',
            fontsize=16, weight='bold',
            bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.3))

    plt.tight_layout()
    if save_as:
        fig.savefig(save_as, dpi=300, bbox_inches='tight')
    plt.show()

def plot_publication_clusters(
    df,
    clear_performative,
    clear_expressive,
    pure_technical_coding_clusters,
    scaled_creative_workhorse_clusters,
    cluster_col='cluster',
    figsize=(12, 10),
    pt_size=12,
    alpha_bg=0.08,
    alpha_fg=0.85,
    save_as=None,
    title='UMAP Projection – Publication Quality',
):
    """
    Plots clusters for publication with four custom color groups and legend.
    Colors:
      - clear_performative: light blue (#87cefa)
      - clear_expressive: light salmon (#ffa07a)
      - pure_technical_coding_clusters: dark blue (#1e3a5c)
      - scaled_creative_workhorse_clusters: dark orange (#d2691e)
    If a cluster is in both a clear and a dark group, the dark color takes precedence.
    Legend only shows these four, no count text, and title is customizable.
    """
    # Color and label mapping
    color_map = {}
    label_map = {}
    # Dark colors take precedence
    for cid in pure_technical_coding_clusters:
        color_map[cid] = '#1e3a5c'
        label_map[cid] = 'Pure Technical Coding'
    for cid in scaled_creative_workhorse_clusters:
        color_map[cid] = '#d2691e'
        label_map[cid] = 'Creative Workhorse'
    for cid in clear_performative:
        if cid not in color_map:
            color_map[cid] = '#87cefa'
            label_map[cid] = 'Clear Performative'
    for cid in clear_expressive:
        if cid not in color_map:
            color_map[cid] = '#ffa07a'
            label_map[cid] = 'Clear Expressive'

    # For plotting, get all clusters in any group
    all_clusters = set(clear_performative) | set(clear_expressive) | set(pure_technical_coding_clusters) | set(scaled_creative_workhorse_clusters)
    highlight_ids = sorted(all_clusters)

    fig, ax = plt.subplots(figsize=figsize)

    # Background: all points in lightgrey
    ax.scatter(df['umap_x'], df['umap_y'],
               color='lightgrey', s=pt_size,
               alpha=alpha_bg, linewidths=0)

    # Plot each group, but only one legend entry per group
    legend_labels = {
        'Clear Performative': {'color': '#87cefa', 'shown': False},
        'Clear Expressive': {'color': '#ffa07a', 'shown': False},
        'Pure Technical Coding': {'color': '#1e3a5c', 'shown': False},
        'Creative Workhorse': {'color': '#d2691e', 'shown': False},
    }

    for cid in highlight_ids:
        mask = df[cluster_col] == cid
        color = color_map[cid]
        label = label_map[cid]
        show_label = not legend_labels[label]['shown']
        ax.scatter(df.loc[mask, 'umap_x'],
                   df.loc[mask, 'umap_y'],
                   color=color, s=pt_size,
                   alpha=alpha_fg, label=label if show_label else None,
                   linewidths=0)
        legend_labels[label]['shown'] = True

    # Build custom legend handles (one per group)
    handles = [Patch(facecolor=info['color'], label=label) for label, info in legend_labels.items()]
    ax.legend(handles=handles, framealpha=0.9, fontsize=14, title_fontsize=16, loc='upper right')

    # Axis & title styling
    ax.set_xlabel('UMAP Dim 1', fontsize=16)
    ax.set_ylabel('UMAP Dim 2', fontsize=16)
    ax.set_title(title, fontsize=24, pad=15, fontweight='bold')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

    plt.tight_layout()
    if save_as:
        fig.savefig(save_as, dpi=300, bbox_inches='tight')
    plt.show() 