"""

K-Nearest Neighbors Graph Construction
Builds KNN graphs using cuML on GPU

"""

import cupy as cp
from gpu_setup import SEED
import cupyx.scipy.sparse as cpx_sp

def build_knn_gpu(Z, k=15, metric="euclidean"):
    """Build KNN graph on GPU"""

    from cuml.neighbors import NearestNeighbors
    
    # Ensure data is on GPU and contiguous
    Z = cp.asarray(Z, dtype=cp.float32)
    if not Z.flags.c_contiguous:
        Z = cp.ascontiguousarray(Z)
    
    print(f"[KNN] Building graph: n={Z.shape[0]}, dims={Z.shape[1]}, k={k}, metric={metric}")
    
    # Fit nearest neighbors
    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(Z)
    D, I = nn.kneighbors(Z)
    
    # Ensure correct dtypes
    D = D.astype(cp.float32)
    I = I.astype(cp.int32)
    
    # Check if self is first neighbor (sanity check)
    n = Z.shape[0]
    arange_idx = cp.arange(n, dtype=cp.int32)
    self_hits = int(cp.count_nonzero(I[:, 0] == arange_idx))
    includes_self = (self_hits >= int(0.95 * n))
    
    print(f"[KNN] n={n}, k={k}, metric={metric}, includes_self={includes_self}")
    return D, I, includes_self

def store_knn_results(adata, label, D, I, k, metric, includes_self):
    """Store KNN results in adata.uns"""
    adata.uns[f"nn_{label}"] = {
        "distances": D,
        "indices": I,
        "k": k,
        "metric": metric,
        "includes_self": includes_self,
        "seed": SEED
    }
    
    print(f"[STORE] KNN results stored for {label}")

# CHANGED TO FIX KEY ERROR WITH OBSP['CONNECTIVITIES']
def build_knn_graphs_for_mdata(mdata, k=15, metric="euclidean"):
    """Build KNN graphs for RNA and ATAC and store in obsp['connectivities']"""
    
    for mod in ["rna", "atac"]:
        Z = mdata[mod].obsm[f"X_pca_{mod}"]
        D, I, self_hits = build_knn_gpu(Z, k=k, metric=metric)
        
        n_cells = Z.shape[0]
        # Convert distances to similarities (optional, exponential kernel)
        sim = cp.exp(-D)
        rows = cp.repeat(cp.arange(n_cells), k)
        cols = I.ravel()
        data = sim.ravel()
        
        # Build CSR adjacency matrix
        A = cpx_sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
        mdata[mod].obsp["connectivities"] = A
        
        # Keep KNN info in uns for reference
        mdata[mod].uns[f"nn_{mod}"] = {
            "distances": D,
            "indices": I,
            "k": k,
            "metric": metric,
            "includes_self": self_hits
        }
    
    # Summary info
    mdata.uns["knn_summary"] = {
        "k": k,
        "metrics": {"rna": metric, "atac": metric},
        "includes_self": {"rna": mdata["rna"].uns["nn_rna"]["includes_self"],
                          "atac": mdata["atac"].uns["nn_atac"]["includes_self"]}
    }
    
def validate_knn_results(adata, label, expected_k):
    """Validate KNN results"""
    nn_data = adata.uns[f"nn_{label}"]
    D = nn_data["distances"]
    I = nn_data["indices"]
    
    # Check shapes
    n_cells = adata.n_obs
    assert D.shape == (n_cells, expected_k), f"Wrong distance shape: {D.shape}"
    assert I.shape == (n_cells, expected_k), f"Wrong indices shape: {I.shape}"
    
    # Check data validity
    assert cp.isfinite(D).all(), "Non-finite distances"
    assert (D >= 0).all(), "Negative distances"
    assert (I >= 0).all() and (I < n_cells).all(), "Invalid indices"
    
    print(f"[VALIDATE] KNN {label}: OK")

if __name__ == "__main__":
    print("KNN graph module ready for import")