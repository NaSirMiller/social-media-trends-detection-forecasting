import math
import polars as pl
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def _build_keyword_freq(keyword_strings: list[str]) -> dict[str, int]:
    counter = Counter()
    for s in keyword_strings:
        if s:
            counter.update(k.strip() for k in s.split(","))
    return dict(counter)

def plot_wordclouds(sample: pl.DataFrame, cluster_col: str = "cluster_tfidf"):
    clusters = sorted(sample[cluster_col].unique().to_list())
    n_cols = 6
    n_rows = math.ceil(len(clusters) / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 4, n_rows * 3)
    )
    axes = axes.flatten()

    for i, cid in enumerate(clusters):
        keywords = (
            sample
            .filter(pl.col(cluster_col) == cid)["english_keywords"]
            .drop_nulls()
            .to_list()
        )
        freq = _build_keyword_freq(keywords)
        if not freq:
            axes[i].axis("off")
            continue

        wc = WordCloud(
            width=400, height=300,
            background_color="white",
            max_words=40,
            colormap="viridis",
        ).generate_from_frequencies(freq)

        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].set_title(f"Cluster {cid}", fontsize=10, pad=4)
        axes[i].axis("off")

    # hide unused axes
    for j in range(len(clusters), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Top keywords per cluster", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("wordclouds.png", dpi=150, bbox_inches="tight")
    plt.show()
    
def plot_sentiment_boxplot(sample: pl.DataFrame, cluster_col: str = "cluster_tfidf"):
    clusters = sorted(sample[cluster_col].unique().to_list())

    # build list of sentiment arrays w/ one per cluster
    data = [
        sample
        .filter(pl.col(cluster_col) == cid)["sentiment"]
        .drop_nulls()
        .to_numpy()
        for cid in clusters
    ]

    fig, ax = plt.subplots(figsize=(18, 5))

    bp = ax.boxplot(
        data,
        patch_artist=True,
        medianprops=dict(color="#D85A30", linewidth=1.5),
        whiskerprops=dict(color="#888780"),
        capprops=dict(color="#888780"),
        flierprops=dict(marker="o", markersize=2, alpha=0.3, color="#888780"),
        boxprops=dict(facecolor="#B5D4F4", color="#185FA5"),
    )

    ax.set_xticks(range(1, len(clusters) + 1))
    ax.set_xticklabels([f"C{c}" for c in clusters], fontsize=8)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Sentiment score")
    ax.set_title("Sentiment distribution per cluster")
    ax.axhline(0, color="#888780", linewidth=0.5, linestyle="--")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    plt.savefig("sentiment_boxplot.png", dpi=150, bbox_inches="tight")
    plt.show()
    
def plot_unique_authors(sample: pl.DataFrame, cluster_col: str = "cluster_tfidf"):
    author_counts = (
        sample
        .group_by(cluster_col)
        .agg(pl.col("author_hash").n_unique().alias("unique_authors"))
        .sort(cluster_col)
    )

    clusters = author_counts[cluster_col].to_list()
    counts   = author_counts["unique_authors"].to_list()

    fig, ax = plt.subplots(figsize=(16, 5))

    bars = ax.bar(
        range(len(clusters)), counts,
        color="#5DCAA5", edgecolor="#0F6E56", linewidth=0.5
    )

    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels([f"C{c}" for c in clusters], fontsize=8)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Unique authors")
    ax.set_title("Unique authors per cluster")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    # annotate bar tops
    for bar, v in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(v), ha="center", va="bottom", fontsize=7
        )

    plt.tight_layout()
    plt.savefig("unique_authors.png", dpi=150, bbox_inches="tight")
    plt.show()