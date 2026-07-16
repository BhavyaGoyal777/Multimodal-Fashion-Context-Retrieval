import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from models import AXES, DEFAULT_PARSER_MODEL, ModelLoader


def load_index(index_path):
    """Load exact-search embedding matrices without allowing pickle."""
    with np.load(index_path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def rank_images(
    query,
    query_axes,
    index,
    loader,
    mode="axis",
    global_weight=0.5,
    missing_axis_penalty=0.1,
):
    """Calculate exact cosine scores for every indexed image."""
    if not 0.0 <= global_weight <= 1.0:
        raise ValueError("global_weight must be between 0 and 1")

    global_query = loader.encode_text(query)
    global_scores = index["global_embeddings"] @ global_query

    if mode == "global":
        final_scores = global_scores
    else:
        axis_sum = np.zeros_like(global_scores)
        active_axis_count = 0

        for axis_name, axis_text in query_axes.items():
            if axis_name not in AXES:
                continue

            axis_query = loader.encode_text(axis_text)
            region_scores = index[f"{axis_name}_embeddings"] @ axis_query
            region_present = index[f"{axis_name}_present"]

            axis_sum += np.where(
                region_present,
                region_scores,
                -missing_axis_penalty,
            )
            active_axis_count += 1

        if active_axis_count:
            mean_axis_score = axis_sum / active_axis_count
            final_scores = (
                global_weight * global_scores
                + (1.0 - global_weight) * mean_axis_score
            )
        else:
            print("No valid regional axes found; using global scores.")
            final_scores = global_scores

    ranked_indices = np.argsort(-final_scores)
    return [
        (
            int(image_idx),
            float(final_scores[image_idx]),
            str(index["paths"][image_idx]),
        )
        for image_idx in ranked_indices
    ]


def retrieve_and_display(
    query,
    query_axes,
    index_path,
    top_k=10,
    mode="axis",
    global_weight=0.5,
    use_parser=False,
    parser_model=DEFAULT_PARSER_MODEL,
    output_path="retrieval_results.png",
):
    """Retrieve and display results"""

    print(f"\n{'='*60}")
    print("FASHION IMAGE RETRIEVAL")
    print(f"{'='*60}\n")

    loader = ModelLoader()
    loader.load_clip_model()

    if use_parser and mode == "axis":
        loader.load_query_parser(parser_model)
        print(f"Original query: '{query}'\n")
        query_axes = loader.parse_query(query)
    else:
        print(f"Query: '{query}'")
        print(f"Query axes: {query_axes}\n")

    index = load_index(index_path)
    image_count = len(index["paths"])
    print(f"Loaded index: {image_count} images")
    print(f"Mode: {mode} exact cosine search\n")

    results = rank_images(
        query=query,
        query_axes=query_axes,
        index=index,
        loader=loader,
        mode=mode,
        global_weight=global_weight,
    )[:min(top_k, image_count)]

    print(f"\nTop {len(results)} results:")
    print(f"{'='*60}")
    for i, (idx, score, path) in enumerate(results, 1):
        print(f"{i}. Score: {score:.3f} - {path}")
    print(f"{'='*60}\n")

    display_grid(results, query, mode, output_path)


def display_grid(
    results,
    query,
    mode,
    output_path,
    grid_size=(2, 5),
):
    """Display results in a grid"""
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(20, 8))
    axes = axes.flatten()

    for i, (idx, score, path) in enumerate(results[:rows*cols]):
        try:
            img = Image.open(path)
            axes[i].imshow(img)
            axes[i].set_title(f"Rank {i+1}\nScore: {score:.3f}", fontsize=10)
            axes[i].axis('off')
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error loading\n{path}",
                        ha='center', va='center')
            axes[i].axis('off')

    for i in range(len(results), rows*cols):
        axes[i].axis('off')

    plt.suptitle(
        f"Mode: {mode} | Query: '{query}'",
        fontsize=16,
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved results to {output_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve fashion images")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument("--index_path", type=str, default="./fashion_index.npz", help="Path to NumPy index")
    parser.add_argument("--top_k", type=int, default=10, help="Number of results to return")
    parser.add_argument("--mode", choices=["global", "axis"], default="axis")
    parser.add_argument("--global_weight", type=float, default=0.5)
    parser.add_argument("--use_parser", action="store_true", help="Use local Qwen parser")
    parser.add_argument("--parser_model", type=str, default=DEFAULT_PARSER_MODEL)
    parser.add_argument("--output", type=str, default="retrieval_results.png")
    parser.add_argument("--axes", type=str, default=None,
                       help='Manual axes in JSON format, e.g., \'{"upper": "red shirt", "lower": "blue jeans"}\'')

    args = parser.parse_args()

    if args.axes:
        query_axes = json.loads(args.axes)
    else:
        query_axes = {}

    retrieve_and_display(
        query=args.query,
        query_axes=query_axes,
        index_path=args.index_path,
        top_k=args.top_k,
        mode=args.mode,
        global_weight=args.global_weight,
        use_parser=args.use_parser,
        parser_model=args.parser_model,
        output_path=args.output,
    )
