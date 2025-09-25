import time
import numpy as np
import pandas as pd
import muon as mu
from mudata import MuData
from anndata import AnnData
from pathlib import Path
import scanpy as sc

## (for plotting WNN + UMAP) 
import scipy.sparse as sp
import cupy as cp
import cupyx.scipy.sparse as cpsp
##

from wnn_gpu import wnn_gpu  # Part 1
from main_knn_pipeline import run_person2_pipeline  # Part 2
from _fusion_portable import fuse_from_mudata, backend_name  # Part 3
from clustering import run_part4_clustering # Part 4

try:
    import cudf
    import cupy as cp
    import cugraph
    import cuml
    HAS_CUGRAPH = True
except ImportError:
    HAS_CUGRAPH = False

# -----------------------------
# Function 1: Generate Dummy MuData
# -----------------------------
def generate_dummy_mudata(n_cells=16000, n_rna_features=5000, n_atac_features=1200) -> MuData:
    """Generate a dummy MuData object with random RNA + ATAC matrices."""
    t0 = time.perf_counter()

    X_rna = np.random.rand(n_cells, n_rna_features).astype(np.float32)
    obs_rna = pd.DataFrame(index=[f"cell{i}" for i in range(n_cells)])
    var_rna = pd.DataFrame(index=[f"gene{i}" for i in range(n_rna_features)])
    adata_rna = AnnData(X=X_rna, obs=obs_rna, var=var_rna)

    X_atac = np.random.rand(n_cells, n_atac_features).astype(np.float32)
    obs_atac = pd.DataFrame(index=[f"cell{i}" for i in range(n_cells)])
    var_atac = pd.DataFrame(index=[f"peak{i}" for i in range(n_atac_features)])
    adata_atac = AnnData(X=X_atac, obs=obs_atac, var=var_atac)

    mdata = MuData({"rna": adata_rna, "atac": adata_atac})
    dt = time.perf_counter() - t0

    print(f"Dummy MuData created in {dt:.2f}s: {mdata.n_obs} cells, RNA={mdata['rna'].n_vars}, ATAC={mdata['atac'].n_vars}")
    return mdata

# for using PMBC 3K RNA dataset with dummy ATAC
def generate_pbmc_mudata(dummy_atac_features=100):
    # Load PBMC3k RNA dataset
    adata_rna = sc.datasets.pbmc3k()
    
    n_cells = adata_rna.n_obs
    
    # Create a dummy ATAC layer
    X_atac = np.random.rand(n_cells, dummy_atac_features).astype(np.float32)
    adata_atac = AnnData(X=X_atac, obs=pd.DataFrame(index=adata_rna.obs_names),
                         var=pd.DataFrame(index=[f"peak{i}" for i in range(dummy_atac_features)]))
    
    mdata = MuData({"rna": adata_rna, "atac": adata_atac})
    print(f"PBMC MuData: {mdata.n_obs} cells, RNA={mdata['rna'].n_vars}, ATAC={mdata['atac'].n_vars}")
    return mdata


# FULL PIPELINE --------------------------------------
def run_pipeline_on_mudata(
    mdata: MuData,
    k: int = 30,
    n_components: int = 50,
    output_file: Path = Path("dummy_person2_output_large.h5mu"),
    weight_mode: str = "entropy_inverse",
    temperature: float = 1.0,
    seed: int = 42
):
    timings = {}

    # -----------------------------
    # Part 1: Alignment
    # -----------------------------
    t0 = time.perf_counter()
    mdata = wnn_gpu(mdata, seed=seed)
    timings["part1"] = time.perf_counter() - t0
    print("Part 1 complete")

    # -----------------------------
    # Part 2: PCA + KNN Graphs
    # -----------------------------
    t0 = time.perf_counter()
    mdata_part2 = run_person2_pipeline(
        mdata,
        output_file=output_file,
        k=k,
        n_components=n_components
    )
    timings["part2"] = time.perf_counter() - t0
    print("Part 2 complete")

    # -----------------------------
    # Part 3: WNN Fusion
    # -----------------------------
    t0 = time.perf_counter()
    print("Running Part 3 fusion; Backend:", backend_name())
    fuse_from_mudata(mdata_part2, weight_mode=weight_mode, temperature=temperature)
    timings["part3"] = time.perf_counter() - t0
    print("Part 3 complete")

    # -----------------------------
    # Part 4: GPU clustering (Leiden/Louvain)
    # -----------------------------
    t0 = time.perf_counter()

    n_clusters = run_part4_clustering(
    mdata_part2,
    obs_key="wnn_leiden",
    method="leiden",
    resolution=0.5,
    random_state=42
    )
    timings["part4"] = time.perf_counter() - t0
    print("Part 4 complete")

    # -----------------------------
    # Part 5: cuML UMAP on RNA PCA
    # -----------------------------
    t0 = time.perf_counter()
    if HAS_CUGRAPH:
        X_rna_pca = mdata_part2["rna"].obsm["X_pca_rna"]
        X_rna_pca_gpu = cp.asarray(X_rna_pca)
        umap_model = cuml.UMAP(n_components=2, random_state=seed)
        embedding_gpu = umap_model.fit_transform(X_rna_pca_gpu)
        mdata_part2["rna"].obsm["X_umap_gpu"] = cp.asnumpy(embedding_gpu)
    timings["part5"] = time.perf_counter() - t0
    print("Part 5 complete (cuML UMAP)")
    
    # -----------------------------
    # Log params and timings in .uns
    # -----------------------------
    mdata_part2.uns["pipeline_params"] = {
        "k": k,
        "n_components": n_components,
        "weight_mode": weight_mode,
        "temperature": temperature,
        "seed": seed,
        "n_rna_features": mdata["rna"].n_vars,
        "n_atac_features": mdata["atac"].n_vars,
        "n_cells": mdata.n_obs
    }
    mdata_part2.uns["pipeline_timings"] = timings

    return mdata_part2, timings


# get the fused WNN graph for plotting
def get_fused_wnn(mdata_part2, as_gpu=True):
    """
    Safely extract the fused WNN graph from a MuData object.
    
    Parameters
    ----------
    mdata_part2 : MuData
        Output from your pipeline after fuse_from_mudata.
    as_gpu : bool, default True
        Whether to return a CuPy matrix (GPU) or SciPy sparse (CPU).

    Returns
    -------
    W_wnn : cupyx.scipy.sparse.csr_matrix or scipy.sparse.csr_matrix
        Fused WNN graph.
    """
    W_wnn = None

    # First, check obsp
    if "wnn_connectivities" in mdata_part2.obsp:
        wnn_pa = mdata_part2.obsp["wnn_connectivities"]

        # Handle PairwiseArrays
        if hasattr(wnn_pa, "to_scipy"):
            W_wnn_cpu = wnn_pa.to_scipy()
        # Handle CuPy CSR
        elif "cupyx" in str(type(wnn_pa)):
            W_wnn_cpu = sp.csr_matrix(wnn_pa.get())  # convert to CPU CSR
        # Handle SciPy CSR
        elif sp.issparse(wnn_pa):
            W_wnn_cpu = wnn_pa
        else:
            raise TypeError(f"Unexpected type in obsp['wnn_connectivities']: {type(wnn_pa)}")

        # Convert to GPU if requested
        if as_gpu:
            W_wnn_gpu = cpsp.csr_matrix(W_wnn_cpu)
            return W_wnn_gpu
        else:
            return W_wnn_cpu

    # Fallback: check .uns if some backends store it there
    elif "wnn_graph" in mdata_part2.uns:
        W_wnn_cpu = mdata_part2.uns["wnn_graph"]
        if as_gpu:
            W_wnn_gpu = cpsp.csr_matrix(W_wnn_cpu)
            return W_wnn_gpu
        else:
            return W_wnn_cpu

    else:
        raise KeyError("Fused WNN graph not found in obsp['wnn_connectivities'] or uns['wnn_graph']")



if __name__ == "__main__":
    # Example 1: Dummy MuData
    # mdata = generate_dummy_mudata(n_cells=16000, n_rna_features=1000, n_atac_features=1200)

    # Example 2: PBMC 3K RNA dataset + dummy ATAC
    mdata = generate_pbmc_mudata(dummy_atac_features=100)

    # Run the full pipeline
    mdata_final, timings = run_pipeline_on_mudata(
        mdata,
        k=30,
        n_components=50
    )

    print("\n--- Step timings (seconds) ---")
    for step, t in timings.items():
        print(f"{step}: {t:.4f}s")

    # Optional: extract fused WNN graph (CPU)
    try:
        W_wnn_cpu = get_fused_wnn(mdata_final, as_gpu=False)
        print(f"Fused WNN graph (CPU) shape: {W_wnn_cpu.shape}")
    except KeyError:
        print("Fused WNN graph not found.")
