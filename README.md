# Fashion Multimodal Retrieval

Exact compositional fashion retrieval using FashionCLIP and SegFormer.

## Architecture

```
Image ── FashionCLIP ───────────────────────────→ Global embedding
   └── SegFormer ── masked regions ── FashionCLIP
                                      ├─────────→ Upper embedding
                                      ├─────────→ Lower embedding
                                      └─────────→ Background embedding

Query ── Qwen parser ── upper/lower/background descriptions
   └── FashionCLIP ── exact cosine scoring over every image ──→ Top-k
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Index Images

```bash
python index.py \
  --image_dir ./images \
  --index_path ./fashion_index.npz \
  --batch_size 32
```

The index contains four normalized embeddings per image: global, upper, lower,
and background.

All visible CUDA GPUs are detected automatically. With Kaggle's two T4 GPUs,
the batch is split across both GPUs. `--batch_size` is the total batch size, not
the per-GPU batch size. Reduce it if CUDA runs out of memory.

### 2. Retrieve Images

**FashionCLIP-only baseline**
```bash
python retrieve.py \
  --query "blue shirt sitting on a park bench" \
  --mode global \
  --output baseline.png
```

**Axis-aware retrieval with local parsing**
```bash
python retrieve.py \
  --query "blue shirt sitting on a park bench" \
  --mode axis \
  --use_parser \
  --output axis_results.png
```

**Axis-aware retrieval with manual parsing**
```bash
python retrieve.py \
  --query "blue shirt with black pants" \
  --mode axis \
  --axes '{"upper":"blue shirt","lower":"black pants"}'
```

## Axes

- `upper`: Upper body clothing (shirts, tops, jackets)
- `lower`: Lower body clothing (pants, jeans, skirts)
- `background`: Scene/environment

## Parameters

**Indexing:**
- `--image_dir`: Directory with images
- `--index_path`: Output NumPy archive
- `--batch_size`: Total inference batch across all GPUs

**Retrieval:**
- `--query`: Search query
- `--axes`: Manual JSON axis specification
- `--use_parser`: Parse with local Qwen
- `--parser_model`: Hugging Face parser model
- `--mode`: `global` baseline or `axis`
- `--global_weight`: Global contribution in axis mode
- `--top_k`: Number of results (default: 10)
- `--index_path`: NumPy index path
- `--output`: Result-grid path
