# GPU Weighted Nearest Neighbors (WNN) Multi-Omics Pipeline

This project implements a **multi-omics analysis pipeline** for single-cell RNA and ATAC data using Weighted Nearest Neighbors (WNN). The pipeline integrates RNA and ATAC modalities, performs PCA, constructs kNN and WNN graphs, clusters cells, and generates UMAP embeddings. GPU acceleration via **cuML** is supported for faster computation.

---

## Features

- Perform **RNA/ATAC alignment** 
- Compute **PCA and kNN graphs** per modality 
- Fuse modality-specific graphs into a **Weighted Nearest Neighbor (WNN) graph** 
- Perform **GPU clustering** using Leiden/Louvain algorithm 
- Generate **UMAP embeddings** on RNA PCA
- Extract and visualize fused WNN graph for downstream analysis.

---

## Pipeline Overview

1. **Alignment**  
   - Preprocess RNA and ATAC data.  
   - Align modalities and compute initial modality-specific similarities.

2. **PCA and kNN Graphs**  
   - Compute PCA on RNA and ATAC separately.  
   - Build k-nearest neighbor graphs per modality.

3. **WNN Fusion**  
   - Fuse RNA and ATAC kNN graphs into a single WNN graph.  
   - Weighted edges reflect relative contribution of each modality.

4. **Clustering**  
   - Cluster cells using GPU-accelerated Leiden or Louvain algorithm.  
   - Adds cluster labels to `mdata.obs['wnn_leiden']`.

5. **UMAP Embedding**  
   - Embed RNA PCA into 2D space using UMAP.  
   - Embeddings stored in `mdata['rna'].obsm['X_umap_gpu']`.

---

## Example Usage

```python
from pipeline import generate_pbmc_mudata, run_pipeline_on_mudata

# Load PBMC RNA data with dummy ATAC
mdata = generate_pbmc_mudata(dummy_atac_features=100)

# Run full WNN pipeline
mdata_final, timings = run_pipeline_on_mudata(mdata, k=30, n_components=50)

# Print step timings
for step, t in timings.items():
    print(f"{step}: {t:.2f}s")

```

---

### Accessing Fused WNN Graph

The Fused WNN Graph can be retrieved using: 

```python
from wnn_pipeline import get_fused_wnn

# CPU sparse matrix
W_wnn_cpu = get_fused_wnn(mdata_final, as_gpu=False)

# GPU CuPy sparse matrix
W_wnn_gpu = get_fused_wnn(mdata_final, as_gpu=True)
```


