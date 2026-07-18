# Multimodal Fashion and Context Retrieval

A training-free fashion image retrieval system that combines FashionCLIP,
SegFormer, and structured query decomposition. It supports both a global
FashionCLIP baseline and a fine-grained retrieval mode that scores upper
clothing, lower clothing, and background context separately.

The current implementation uses exact cosine search over a NumPy index. This
keeps the ML comparison simple and avoids approximate-search effects on the
3,200-image experiment.

## Method overview

```text
Input image
   |
   +-- complete image ----------------------> FashionCLIP --> global vector
   |
   +-- SegFormer clothing segmentation
          |
          +-- upper-clothing masked crop ---> FashionCLIP --> upper vector
          +-- lower-clothing masked crop ---> FashionCLIP --> lower vector
          +-- background masked crop --------> FashionCLIP --> background vector

Natural-language query
   |
   +-- complete query ----------------------> FashionCLIP --> global query vector
   |
   +-- optional local Qwen parser
          +-- upper description ------------> FashionCLIP --> upper query vector
          +-- lower description ------------> FashionCLIP --> lower query vector
          +-- background description --------> FashionCLIP --> background query vector
```

The indexed segmentation labels are intentionally limited to four image
representations:

| Representation | SegFormer class IDs | Meaning |
|---|---:|---|
| `global` | Not applicable | Complete image |
| `upper` | 4, 7 | Upper clothes and dresses |
| `lower` | 5, 6 | Skirts and pants |
| `background` | 0 | Scene and environment |

Accessories, shoes, pose, identity, and other unsupported details remain
available through the global representation.

## Repository structure

```text
.
|-- models.py                 Model loading, segmentation, region creation,
|                             FashionCLIP encoding, and query parsing
|-- index.py                  Batched image-indexing workflow
|-- retrieve.py               Global and axis-aware exact retrieval workflow
|-- cluster_index.py          Optional embedding inspection and clustering
|-- requirements.txt          Python dependencies
|-- README.md                 Setup, architecture, and usage guide
`-- report/
    |-- main.tex              Assignment report source
    |-- references.bib        BibTeX references
    `-- results_optimized/    Compressed qualitative result grids used by
                              the report
```

### `models.py`

Contains the shared ML logic used by indexing and retrieval:

- Detects CUDA and uses `cuda:0` when available, otherwise CPU.
- Loads `mattmdjaga/segformer_b2_clothes` for semantic segmentation.
- Loads `Marqo/marqo-fashionCLIP` through OpenCLIP.
- Segments a batch of images and resizes masks to the original resolution.
- Creates masked, padded regional crops with unrelated pixels replaced by
  neutral grey.
- Produces L2-normalized FashionCLIP image and text vectors.
- Optionally loads a local Hugging Face causal language model and parses a
  query into `upper`, `lower`, and `background` JSON fields.

The default parser is `Qwen/Qwen2.5-1.5B-Instruct`. A different compatible
model can be selected with `--parser_model`.

### `index.py`

Implements Part A, the indexer:

1. Recursively discovers JPG, JPEG, and PNG files.
2. Loads source images in batches.
3. Runs batched SegFormer inference.
4. Creates upper, lower, and background crops when those regions are present.
5. Places global and regional views into a shared FashionCLIP encoding queue.
6. Saves normalized vectors, image paths, and region-presence flags to one
   `.npz` archive.

`--batch_size` controls the SegFormer batch size. `--clip_batch_size` controls
the larger global-plus-regional FashionCLIP queue. The implementation uses one
GPU intentionally; additional visible GPUs are reported but not used.

### `retrieve.py`

Implements Part B, the retriever:

- Loads the `.npz` index without pickle.
- Encodes the complete query for global retrieval.
- Accepts manually supplied axes or uses a local Qwen model to parse them.
- Scores every indexed image using exact normalized dot products, equivalent
  to cosine similarity.
- Returns ranked paths and scores.
- Saves the top results as a visualization grid.

Two retrieval modes are available:

- `global`: FashionCLIP-only baseline using the complete image and query.
- `axis`: combines the global score with the mean score of requested regional
  axes.

For axis mode, the score is calculated as:

```text
axis_mean  = mean(score for each requested regional axis)
final      = global_weight * global_score
             + (1 - global_weight) * axis_mean
```

If the query requests a region that SegFormer did not find in an image, that
axis receives a small negative score of `-0.1` instead of being silently
ignored. If no valid axes are supplied, retrieval falls back to global mode.

### `cluster_index.py`

Provides optional, label-free inspection of the indexed feature spaces. It
clusters the global, upper, lower, and background embeddings independently
using MiniBatchKMeans.

When `--clusters` is omitted, the script evaluates the supplied candidate
values of `k` and selects the one with the highest cosine silhouette score for
each representation. It produces:

- `cluster_assignments.csv`: cluster assignment and centroid distance for
  every available image representation.
- `cluster_selection.csv`: silhouette score for each tested value of `k`.
- `cluster_selection.png`: cluster-count comparison plot.
- One folder of representative cluster grids per selected representation.

Clustering is used to inspect dataset coverage and find useful qualitative
queries. It is not treated as retrieval ground truth or a precision metric.

### `report/`

Contains the final LaTeX report, references, and optimized qualitative result
figures. The optimized images are smaller copies used to avoid Overleaf
compile-time limits; they do not change retrieval rankings.

## Installation

Python 3.10 or newer is recommended. A CUDA GPU is strongly recommended for
indexing.

```bash
pip install -r requirements.txt
```

The first run downloads the FashionCLIP, SegFormer, and optional parser model
from Hugging Face. An internet connection or populated model cache is therefore
required.

## Running the pipeline

### 1. Build the index

Place the dataset in a directory such as `images/`, then run:

```bash
python index.py \
  --image_dir ./images \
  --index_path ./fashion_index.npz \
  --batch_size 32 \
  --clip_batch_size 128
```

For a T4 GPU, `32` and `128` are useful starting values. If indexing runs out
of CUDA memory, reduce both values:

```bash
python index.py \
  --image_dir ./images \
  --index_path ./fashion_index.npz \
  --batch_size 16 \
  --clip_batch_size 64
```

The resulting archive contains:

```text
paths
global_embeddings
upper_embeddings
upper_present
lower_embeddings
lower_present
background_embeddings
background_present
```

Keep the indexed images at paths accessible to the retrieval environment,
because result visualization opens the paths stored in the archive.

### 2. Run the FashionCLIP baseline

```bash
mkdir -p results

python retrieve.py \
  --query "A pink top with a black skirt on an urban street" \
  --index_path ./fashion_index.npz \
  --mode global \
  --top_k 5 \
  --output ./results/pink_black_street_global_top5.png
```

The global baseline does not load the language-model parser.

### 3. Run axis-aware retrieval with automatic parsing

```bash
python retrieve.py \
  --query "A pink top with a black skirt on an urban street" \
  --index_path ./fashion_index.npz \
  --mode axis \
  --use_parser \
  --global_weight 0.6 \
  --top_k 5 \
  --output ./results/pink_black_street_axis_top5.png
```

The parser should produce fields similar to:

```json
{
  "upper": "pink top",
  "lower": "black skirt",
  "background": "urban street"
}
```

### 4. Run axis-aware retrieval with manual axes

Manual axes are useful for evaluating retrieval independently of parser
quality:

```bash
python retrieve.py \
  --query "A pink top with a black skirt on an urban street" \
  --index_path ./fashion_index.npz \
  --mode axis \
  --axes '{"upper":"pink top","lower":"black skirt","background":"urban street"}' \
  --global_weight 0.6 \
  --top_k 5 \
  --output ./results/pink_black_street_manual_axis_top5.png
```

### 5. Inspect embedding clusters

Run this after building the index:

```bash
python cluster_index.py \
  --index_path ./fashion_index.npz \
  --output_dir ./clusters \
  --representations global upper lower background \
  --k_candidates 5 10 15 20 25 30 \
  --silhouette_sample_size 1000 \
  --samples_per_cluster 12
```

To bypass automatic silhouette selection, provide a fixed cluster count:

```bash
python cluster_index.py \
  --index_path ./fashion_index.npz \
  --output_dir ./clusters \
  --clusters 20
```

## Processing many queries efficiently

Every separate `python retrieve.py` command starts a new process and reloads
FashionCLIP and the optional parser. For many queries, import the retrieval
functions and keep the models on the GPU in one Python process:

```python
from models import ModelLoader, DEFAULT_PARSER_MODEL
from retrieve import load_index, rank_images

index = load_index("fashion_index.npz")

loader = ModelLoader()
loader.load_clip_model()
loader.load_query_parser(DEFAULT_PARSER_MODEL)

queries = [
    "A pink top with a black skirt on an urban street",
    "A blue shirt with beige pants in a green park",
]

for query in queries:
    axes = loader.parse_query(query)
    results = rank_images(
        query=query,
        query_axes=axes,
        index=index,
        loader=loader,
        mode="axis",
        global_weight=0.6,
    )[:5]
    print(query, results)
```

## Evaluation

The dataset does not provide query-level relevance labels, so the current
submission does not claim precision, recall, or mAP. Evaluation is presented
as a qualitative global-versus-axis comparison across upper-only, lower-only,
background-only, compositional, and three-axis queries.

A future labelled benchmark should report metrics such as Recall@K,
Precision@K, mAP, and nDCG, with particular attention to colour-to-garment
binding and scene context.

## Current limitations

- Retrieval quality depends on SegFormer mask quality and its coarse clothing
  label set.
- Ties, bags, shoes, and other unsupported accessories rely mainly on global
  FashionCLIP evidence.
- Grey masked crops differ from FashionCLIP's original training distribution.
- Query parsing can omit or misassign attributes.
- Fixed axis weights are not optimal for every query.
- Exact search is appropriate for the current dataset but should be replaced
  by ANN candidate retrieval and exact reranking at million-image scale.

## Proposed extension

Future work can replace fixed crops with learned semantic axis tokens for the
vision and text encoders. SegFormer masks can provide weak regional
supervision, while a SigLIP-style sigmoid contrastive objective and hard
attribute-swapped negatives can train explicit upper, lower, and background
representations. This extension requires training and is not part of the
current implementation.
