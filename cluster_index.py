import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm


REPRESENTATIONS = ("global", "upper", "lower", "background")


def load_index(index_path):
    """Load all embedding matrices from the NumPy index."""
    with np.load(index_path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def get_representation(index, representation):
    """Return paths, embeddings and original indices for one representation."""
    paths = index["paths"]

    if representation == "global":
        present = np.ones(len(paths), dtype=bool)
        embeddings = index["global_embeddings"]
    else:
        present = index[f"{representation}_present"]
        embeddings = index[f"{representation}_embeddings"]

    original_indices = np.flatnonzero(present)
    return (
        paths[present],
        embeddings[present].astype(np.float32),
        original_indices,
    )


def fit_kmeans(embeddings, cluster_count, random_state):
    """Fit MiniBatchKMeans to one normalized embedding matrix."""
    model = MiniBatchKMeans(
        n_clusters=cluster_count,
        batch_size=min(256, len(embeddings)),
        n_init=10,
        random_state=random_state,
    )
    labels = model.fit_predict(embeddings)
    return model, labels


def select_cluster_count(
    embeddings,
    candidates,
    sample_size,
    random_state,
):
    """Select k using the highest cosine silhouette score."""
    valid_candidates = sorted({
        candidate
        for candidate in candidates
        if 2 <= candidate < len(embeddings)
    })

    if not valid_candidates:
        raise ValueError(
            "No valid k candidates. Every k must be at least 2 and smaller "
            "than the number of available embeddings."
        )

    evaluated = []
    best_result = None

    for cluster_count in tqdm(
        valid_candidates,
        desc="Selecting cluster count",
        leave=False,
    ):
        model, labels = fit_kmeans(
            embeddings,
            cluster_count,
            random_state,
        )

        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            score = -1.0
        else:
            actual_sample_size = min(sample_size, len(embeddings))
            score_kwargs = {}
            if actual_sample_size < len(embeddings):
                score_kwargs = {
                    "sample_size": actual_sample_size,
                    "random_state": random_state,
                }

            score = float(silhouette_score(
                embeddings,
                labels,
                metric="cosine",
                **score_kwargs,
            ))

        result = {
            "k": cluster_count,
            "silhouette": score,
            "model": model,
            "labels": labels,
        }
        evaluated.append(result)

        if best_result is None or score > best_result["silhouette"]:
            best_result = result

    return best_result, evaluated


def save_cluster_grid(
    paths,
    labels,
    distances,
    cluster_id,
    output_path,
    samples_per_cluster,
):
    """Save the images nearest to one cluster centroid as a contact sheet."""
    members = np.flatnonzero(labels == cluster_id)
    members = members[np.argsort(distances[members])]
    members = members[:samples_per_cluster]

    columns = min(4, max(1, len(members)))
    rows = max(1, math.ceil(len(members) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4 * columns, 4 * rows),
        squeeze=False,
    )
    axes = axes.flatten()

    for axis, member_idx in zip(axes, members):
        image_path = str(paths[member_idx])
        try:
            with Image.open(image_path) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_title(
                f"{Path(image_path).name}\n"
                f"distance={distances[member_idx]:.3f}",
                fontsize=8,
            )
        except Exception as error:
            axis.text(
                0.5,
                0.5,
                f"Could not load\n{Path(image_path).name}\n{error}",
                ha="center",
                va="center",
            )
        axis.axis("off")

    for axis in axes[len(members):]:
        axis.axis("off")

    figure.suptitle(f"Cluster {cluster_id}", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def cluster_representation(
    index,
    representation,
    cluster_count,
    k_candidates,
    silhouette_sample_size,
    samples_per_cluster,
    output_dir,
    random_state,
):
    """Cluster one embedding representation and save assignments and grids."""
    paths, embeddings, original_indices = get_representation(
        index,
        representation,
    )

    if len(paths) == 0:
        print(f"Skipping {representation}: no embeddings are present")
        return [], []

    selection_rows = []

    if cluster_count is None:
        best_result, evaluated = select_cluster_count(
            embeddings=embeddings,
            candidates=k_candidates,
            sample_size=silhouette_sample_size,
            random_state=random_state,
        )
        actual_cluster_count = best_result["k"]
        model = best_result["model"]
        labels = best_result["labels"]

        for result in evaluated:
            selection_rows.append({
                "axis": representation,
                "k": result["k"],
                "silhouette": result["silhouette"],
                "selected": result["k"] == actual_cluster_count,
            })

        print(
            f"{representation}: selected k={actual_cluster_count} "
            f"(silhouette={best_result['silhouette']:.4f})"
        )
    else:
        actual_cluster_count = min(cluster_count, len(paths))
        if actual_cluster_count < 2:
            raise ValueError("clusters must be at least 2")
        model, labels = fit_kmeans(
            embeddings,
            actual_cluster_count,
            random_state,
        )
        print(f"{representation}: using manual k={actual_cluster_count}")

    distances = np.linalg.norm(
        embeddings - model.cluster_centers_[labels],
        axis=1,
    )

    representation_dir = output_dir / representation
    representation_dir.mkdir(parents=True, exist_ok=True)

    for cluster_id in tqdm(
        range(actual_cluster_count),
        desc=f"Saving {representation} clusters",
    ):
        save_cluster_grid(
            paths=paths,
            labels=labels,
            distances=distances,
            cluster_id=cluster_id,
            output_path=representation_dir / f"cluster_{cluster_id:02d}.png",
            samples_per_cluster=samples_per_cluster,
        )

    rows = []
    for local_idx, image_path in enumerate(paths):
        rows.append({
            "axis": representation,
            "image_index": int(original_indices[local_idx]),
            "path": str(image_path),
            "cluster": int(labels[local_idx]),
            "distance_to_centroid": float(distances[local_idx]),
        })
    return rows, selection_rows


def save_selection_results(selection_rows, output_dir):
    """Save silhouette scores as a CSV file and line plot."""
    if not selection_rows:
        return

    selection_path = output_dir / "cluster_selection.csv"
    with selection_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["axis", "k", "silhouette", "selected"],
        )
        writer.writeheader()
        writer.writerows(selection_rows)

    figure, axis = plt.subplots(figsize=(10, 6))
    represented_axes = sorted({row["axis"] for row in selection_rows})

    for representation in represented_axes:
        representation_rows = [
            row for row in selection_rows
            if row["axis"] == representation
        ]
        representation_rows.sort(key=lambda row: row["k"])

        axis.plot(
            [row["k"] for row in representation_rows],
            [row["silhouette"] for row in representation_rows],
            marker="o",
            label=representation,
        )

    axis.set_xlabel("Number of clusters (k)")
    axis.set_ylabel("Cosine silhouette score")
    axis.set_title("Cluster-count selection")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_dir / "cluster_selection.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"Cluster-selection scores saved to: {selection_path}")


def cluster_index(
    index_path,
    output_dir,
    representations,
    cluster_count=None,
    k_candidates=(5, 10, 15, 20, 25, 30),
    silhouette_sample_size=1000,
    samples_per_cluster=12,
    random_state=42,
):
    """Cluster selected representations from an existing image index."""
    index = load_index(index_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assignment_rows = []
    selection_rows = []
    for representation in representations:
        representation_rows, representation_selection = cluster_representation(
            index=index,
            representation=representation,
            cluster_count=cluster_count,
            k_candidates=k_candidates,
            silhouette_sample_size=silhouette_sample_size,
            samples_per_cluster=samples_per_cluster,
            output_dir=output_dir,
            random_state=random_state,
        )
        assignment_rows.extend(representation_rows)
        selection_rows.extend(representation_selection)

    assignments_path = output_dir / "cluster_assignments.csv"
    with assignments_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "axis",
                "image_index",
                "path",
                "cluster",
                "distance_to_centroid",
            ],
        )
        writer.writeheader()
        writer.writerows(assignment_rows)

    save_selection_results(selection_rows, output_dir)

    print(f"Cluster assignments saved to: {assignments_path}")
    print(f"Cluster grids saved under: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster global and regional image embeddings"
    )
    parser.add_argument("--index_path", default="./fashion_index.npz")
    parser.add_argument("--output_dir", default="./clusters")
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATIONS,
        default=list(REPRESENTATIONS),
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=None,
        help="Manual k override. Omit to select k automatically.",
    )
    parser.add_argument(
        "--k_candidates",
        nargs="+",
        type=int,
        default=[5, 10, 15, 20, 25, 30],
        help="Candidate k values used by automatic selection.",
    )
    parser.add_argument(
        "--silhouette_sample_size",
        type=int,
        default=1000,
        help="Maximum samples used to calculate each silhouette score.",
    )
    parser.add_argument("--samples_per_cluster", type=int, default=12)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    cluster_index(
        index_path=args.index_path,
        output_dir=args.output_dir,
        representations=args.representations,
        cluster_count=args.clusters,
        k_candidates=args.k_candidates,
        silhouette_sample_size=args.silhouette_sample_size,
        samples_per_cluster=args.samples_per_cluster,
        random_state=args.random_state,
    )
