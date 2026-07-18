import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
from tqdm import tqdm

from models import ModelLoader, AXES


def get_image_files(directory):
    """Return all supported images under a directory."""
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_files = []

    directory = Path(directory)
    for ext in extensions:
        image_files.extend(directory.glob(ext))
        image_files.extend(directory.rglob(ext))

    return sorted(set(image_files))


def index_images(image_dir, index_path, batch_size=32, clip_batch_size=128):
    """Index global, upper, lower and background FashionCLIP embeddings."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if clip_batch_size < 1:
        raise ValueError("clip_batch_size must be at least 1")

    print(f"\n{'='*60}")
    print("FASHION IMAGE INDEXER")
    print(f"{'='*60}\n")

    loader = ModelLoader()
    loader.load_segmentation_model()
    loader.load_clip_model()

    image_files = get_image_files(image_dir)
    print(f"Found {len(image_files)} images in {image_dir}\n")

    if len(image_files) == 0:
        print("No images found. Exiting.")
        return

    print(
        f"Indexing {len(image_files)} images "
        f"(SegFormer batch: {batch_size}, "
        f"FashionCLIP batch: {clip_batch_size})...\n"
    )

    num_indexed = 0
    num_errors = 0
    indexed_paths = []
    global_embeddings = []
    axis_embeddings = {axis_name: [] for axis_name in AXES}
    axis_present = {axis_name: [] for axis_name in AXES}
    timings = {
        "load": 0.0,
        "segmentation": 0.0,
        "region_creation": 0.0,
        "fashionclip": 0.0,
        "save": 0.0,
    }

    progress = tqdm(
        range(0, len(image_files), batch_size),
        desc="Indexing",
    )
    for batch_idx in progress:
        batch_files = image_files[batch_idx:batch_idx + batch_size]

        loaded_files = []
        images = []

        stage_start = perf_counter()
        for img_path in batch_files:
            try:
                with Image.open(img_path) as source:
                    images.append(source.convert("RGB"))
                loaded_files.append(img_path)
            except Exception as e:
                num_errors += 1
                tqdm.write(f"Error loading {img_path.name}: {e}")
                continue
        timings["load"] += perf_counter() - stage_start

        if not images:
            continue

        try:
            stage_start = perf_counter()
            masks = loader.segment_images(images)
            timings["segmentation"] += perf_counter() - stage_start

            stage_start = perf_counter()
            # Every source image contributes one global FashionCLIP input.
            clip_records = [
                (image_idx, "global", image)
                for image_idx, image in enumerate(images)
            ]

            # Add every available regional input to the same encoding queue.
            for image_idx, (image, mask) in enumerate(zip(images, masks)):
                for axis_name, class_ids in AXES.items():
                    region = loader.extract_region_crop(
                        image=image,
                        masks=mask,
                        class_ids=class_ids,
                    )
                    if region is not None:
                        clip_records.append((image_idx, axis_name, region))
            timings["region_creation"] += perf_counter() - stage_start

            encoded = [{} for _ in images]

            # A 32-image source batch produces at most 128 FashionCLIP inputs.
            # Encoding them together keeps the single GPU better utilized and
            # reduces the number of separate FashionCLIP forward passes.
            stage_start = perf_counter()
            for start in range(0, len(clip_records), clip_batch_size):
                record_batch = clip_records[start:start + clip_batch_size]
                record_embeddings = loader.encode_images([
                    image for _, _, image in record_batch
                ])
                for (image_idx, representation, _), embedding in zip(
                    record_batch,
                    record_embeddings,
                ):
                    encoded[image_idx][representation] = embedding.astype(
                        np.float32
                    )
            timings["fashionclip"] += perf_counter() - stage_start

            for img_path, representations in zip(loaded_files, encoded):
                global_emb = representations["global"]
                indexed_paths.append(str(img_path))
                global_embeddings.append(global_emb)

                for axis_name in AXES:
                    region_emb = representations.get(axis_name)
                    is_present = region_emb is not None
                    axis_present[axis_name].append(is_present)
                    axis_embeddings[axis_name].append(
                        region_emb
                        if is_present
                        else np.zeros_like(global_emb, dtype=np.float32)
                    )

            num_indexed += len(loaded_files)
            progress.set_postfix(
                indexed=num_indexed,
                seg=f"{timings['segmentation']:.1f}s",
                clip=f"{timings['fashionclip']:.1f}s",
            )
        except Exception as e:
            num_errors += len(images)
            tqdm.write(
                f"Error processing batch starting at {loaded_files[0].name}: {e}"
            )
            continue

    if not indexed_paths:
        print("No images were successfully indexed.")
        return

    index_path = Path(index_path)
    if index_path.suffix != ".npz":
        index_path = index_path.with_suffix(".npz")
    index_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "paths": np.asarray(indexed_paths),
        "global_embeddings": np.asarray(global_embeddings, dtype=np.float32),
    }
    for axis_name in AXES:
        payload[f"{axis_name}_embeddings"] = np.asarray(
            axis_embeddings[axis_name], dtype=np.float32
        )
        payload[f"{axis_name}_present"] = np.asarray(
            axis_present[axis_name], dtype=bool
        )

    stage_start = perf_counter()
    np.savez(index_path, **payload)
    timings["save"] = perf_counter() - stage_start

    print(f"\n{'='*60}")
    print("INDEXING COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully indexed: {num_indexed} images")
    print(f"Errors: {num_errors}")
    print(f"Stored embeddings per image: global + {', '.join(AXES)}")
    print(f"Index saved to: {index_path}")
    print("\nTiming breakdown:")
    print(f"  Image loading:   {timings['load']:.1f}s")
    print(f"  Segmentation:    {timings['segmentation']:.1f}s")
    print(f"  Region creation: {timings['region_creation']:.1f}s")
    print(f"  FashionCLIP:     {timings['fashionclip']:.1f}s")
    print(f"  Index saving:    {timings['save']:.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index fashion images")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--index_path", type=str, default="./fashion_index.npz", help="Output NumPy index")
    parser.add_argument("--batch_size", type=int, default=32, help="SegFormer batch size")
    parser.add_argument(
        "--clip_batch_size",
        type=int,
        default=128,
        help="FashionCLIP batch size",
    )

    args = parser.parse_args()

    index_images(
        args.image_dir,
        args.index_path,
        args.batch_size,
        args.clip_batch_size,
    )
