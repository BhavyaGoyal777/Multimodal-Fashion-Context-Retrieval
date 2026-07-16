import argparse
from pathlib import Path

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


def index_images(image_dir, index_path, batch_size=32):
    """Index global, upper, lower and background FashionCLIP embeddings."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

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

    print(f"Indexing {len(image_files)} images (batch size: {batch_size})...\n")

    num_indexed = 0
    num_errors = 0
    indexed_paths = []
    global_embeddings = []
    axis_embeddings = {axis_name: [] for axis_name in AXES}
    axis_present = {axis_name: [] for axis_name in AXES}

    for batch_idx in tqdm(range(0, len(image_files), batch_size), desc="Indexing"):
        batch_files = image_files[batch_idx:batch_idx + batch_size]

        loaded_files = []
        images = []

        for img_path in batch_files:
            try:
                with Image.open(img_path) as source:
                    images.append(source.convert("RGB"))
                loaded_files.append(img_path)
            except Exception as e:
                num_errors += 1
                tqdm.write(f"Error loading {img_path.name}: {e}")
                continue

        if not images:
            continue

        try:
            # Both forward passes are batched and DataParallel automatically
            # splits them across all visible CUDA GPUs.
            masks = loader.segment_images(images)
            global_embs = loader.encode_images(images)

            # Create the three regional images for every successfully loaded image.
            crop_records = []
            for image_idx, (image, mask) in enumerate(zip(images, masks)):
                for axis_name, class_ids in AXES.items():
                    crop = loader.extract_region_crop(
                        image=image,
                        masks=mask,
                        class_ids=class_ids,
                    )
                    if crop is not None:
                        crop_records.append((image_idx, axis_name, crop))

            region_embs = [{} for _ in images]

            # Region count can be 3x the image count. Keep each FashionCLIP
            # forward pass bounded by the requested batch size.
            for start in range(0, len(crop_records), batch_size):
                crop_batch = crop_records[start:start + batch_size]
                crop_embs = loader.encode_images([
                    crop for _, _, crop in crop_batch
                ])
                for (image_idx, axis_name, _), emb in zip(crop_batch, crop_embs):
                    region_embs[image_idx][axis_name] = emb.astype(np.float32)

            for img_path, global_emb, regions in zip(
                loaded_files, global_embs, region_embs
            ):
                indexed_paths.append(str(img_path))
                global_embeddings.append(global_emb.astype(np.float32))

                for axis_name in AXES:
                    region_emb = regions.get(axis_name)
                    is_present = region_emb is not None
                    axis_present[axis_name].append(is_present)
                    axis_embeddings[axis_name].append(
                        region_emb
                        if is_present
                        else np.zeros_like(global_emb, dtype=np.float32)
                    )

            num_indexed += len(loaded_files)
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

    np.savez_compressed(index_path, **payload)

    print(f"\n{'='*60}")
    print("INDEXING COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully indexed: {num_indexed} images")
    print(f"Errors: {num_errors}")
    print(f"Stored embeddings per image: global + {', '.join(AXES)}")
    print(f"Index saved to: {index_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index fashion images")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--index_path", type=str, default="./fashion_index.npz", help="Output NumPy index")
    parser.add_argument("--batch_size", type=int, default=32, help="Total batch size split across GPUs")

    args = parser.parse_args()

    index_images(args.image_dir, args.index_path, args.batch_size)
