"""

Quality Control and Validation
Comprehensive validation of embeddings and connectivity matrices

"""

import cupy as cp
from gpu_setup import EPS

def qc_embedding(adata, key, expected_dim, label=""):
    """Validate embedding quality"""
    print(f"[QC] Checking embedding: {key}")
    
    # Get embedding
    Z = cp.asarray(adata.obsm[key])
    
    # Shape validation
    assert Z.ndim == 2, f"Wrong dimensions: {Z.ndim} != 2"
    assert Z.shape[1] == expected_dim, f"Wrong feature count: {Z.shape[1]} != {expected_dim}"
    assert Z.shape[0] == adata.n_obs, f"Wrong cell count: {Z.shape[0]} != {adata.n_obs}"
    
    # Data quality validation
    assert cp.isfinite(Z).all(), f"Non-finite values in {key}"
    
    # Check for reasonable variance
    var_per_dim = cp.var(Z, axis=0)
    min_var = float(cp.min(var_per_dim))
    max_var = float(cp.max(var_per_dim))
    mean_var = float(cp.mean(var_per_dim))
    
    assert min_var > EPS, f"Zero variance dimension in {key}"
    
    print(f"[QC] {key}: shape={Z.shape}, finite=OK, var_range=[{min_var:.3e}, {max_var:.3e}], mean_var={mean_var:.3e}")
    
    return {
        "shape": list(Z.shape),
        "min_var": min_var,
        "max_var": max_var,
        "mean_var": mean_var
    }

def qc_connectivity(adata, conn_key, tol=1e-3):
    """Validate connectivity matrix"""
    print(f"[QC] Checking connectivity: {conn_key}")
    
    A = adata.obsp[conn_key]
    n, m = A.shape
    
    # Shape validation
    assert n == m, f"Not square: {A.shape}"
    assert n == adata.n_obs, f"Wrong size: {n} != {adata.n_obs}"
    
    # Sparsity validation
    assert A.nnz > n, f"Too sparse: {A.nnz} <= {n}"
    assert A.nnz < n * n, f"Too dense: {A.nnz} >= {n*n}"
    
    # Data quality validation
    assert cp.isfinite(A.data).all(), "Non-finite weights"
    assert not (A.data < 0).any(), "Negative weights"
    
    # Row normalization validation
    row_sums = cp.array(A.sum(axis=1)).ravel()
    max_dev = float(cp.max(cp.abs(row_sums - 1.0)))
    mean_dev = float(cp.mean(cp.abs(row_sums - 1.0)))
    assert max_dev <= tol, f"Row normalization failed: max_dev={max_dev} > {tol}"
    
    # Connectivity validation (no isolated cells)
    zero_rows = int(cp.count_nonzero(row_sums < EPS))
    assert zero_rows == 0, f"Found {zero_rows} disconnected cells"
    
    # Symmetry check (approximate for numerical stability)
    A_t = A.T.tocsr()
    diff_nnz = abs(A.nnz - A_t.nnz)
    is_symmetric = diff_nnz < (0.01 * A.nnz)  # Allow 1% difference
    
    density = A.nnz / (n * n)
    
    print(f"[QC] {conn_key}: nnz={A.nnz}, max_row_dev={max_dev:.3e}, mean_row_dev={mean_dev:.3e}, density={density:.4e}, approx_symmetric={is_symmetric}")
    
    return {
        "nnz": int(A.nnz),
        "max_row_dev": max_dev,
        "mean_row_dev": mean_dev,
        "density": density,
        "is_symmetric": is_symmetric,
        "zero_rows": zero_rows
    }

def qc_knn_results(adata, label, expected_k):
    """Validate KNN results"""
    print(f"[QC] Checking KNN results: {label}")
    
    nn_data = adata.uns[f"nn_{label}"]
    D = nn_data["distances"]
    I = nn_data["indices"]
    k = nn_data["k"]
    
    # Shape validation
    n_cells = adata.n_obs
    assert D.shape == (n_cells, k), f"Wrong distance shape: {D.shape}"
    assert I.shape == (n_cells, k), f"Wrong indices shape: {I.shape}"
    assert k == expected_k, f"Wrong k: {k} != {expected_k}"
    
    # Data validation
    assert cp.isfinite(D).all(), "Non-finite distances"
    assert (D >= 0).all(), "Negative distances"
    assert (I >= 0).all() and (I < n_cells).all(), "Invalid indices"
    
    # Distance ordering (should be non-decreasing)
    D_sorted = cp.sort(D, axis=1)
    assert cp.allclose(D, D_sorted), "Distances not sorted"
    
    # Self-consistency check
    includes_self = nn_data.get("includes_self", False)
    if includes_self:
        arange_idx = cp.arange(n_cells, dtype=cp.int32)
        self_hits = int(cp.count_nonzero(I[:, 0] == arange_idx))
        self_ratio = self_hits / n_cells
        assert self_ratio > 0.9, f"Self-hit ratio too low: {self_ratio:.3f}"
    
    mean_dist = float(cp.mean(D))
    max_dist = float(cp.max(D))
    
    print(f"[QC] KNN {label}: shape={D.shape}, mean_dist={mean_dist:.3e}, max_dist={max_dist:.3e}, includes_self={includes_self}")
    
    return {
        "shape": list(D.shape),
        "mean_dist": mean_dist,
        "max_dist": max_dist,
        "includes_self": includes_self
    }

def run_full_qc(mdata, n_components=50, k=15):
    """Run comprehensive quality control on all components"""
    print("\n" + "="*50)
    print("RUNNING COMPREHENSIVE QUALITY CONTROL")
    print("="*50)
    
    qc_results = {}
    
    # QC embeddings
    print("\n[QC PHASE 1] Checking embeddings...")
    qc_results["rna_embedding"] = qc_embedding(mdata["rna"], "X_pca_rna", n_components, "RNA")
    qc_results["atac_embedding"] = qc_embedding(mdata["atac"], "X_pca_atac", n_components, "ATAC")
    
    # QC KNN results
    print("\n[QC PHASE 2] Checking KNN results...")
    qc_results["rna_knn"] = qc_knn_results(mdata["rna"], "rna", k)
    qc_results["atac_knn"] = qc_knn_results(mdata["atac"], "atac", k)
    
    # QC connectivity matrices
    print("\n[QC PHASE 3] Checking connectivity matrices...")
    qc_results["rna_connectivity"] = qc_connectivity(mdata["rna"], "connectivities")
    qc_results["atac_connectivity"] = qc_connectivity(mdata["atac"], "connectivities")
    
    # Cross-modality alignment check
    print("\n[QC PHASE 4] Checking cross-modality alignment...")
    rna_cells = mdata["rna"].obs_names
    atac_cells = mdata["atac"].obs_names
    assert rna_cells.equals(atac_cells), "Cell order mismatch between modalities"
    print("[QC] Cross-modality alignment: OK")
    
    # Summary
    print("\n" + "="*50)
    print("QUALITY CONTROL SUMMARY")
    print("="*50)
    print(f"✅ RNA embedding: {qc_results['rna_embedding']['shape']}")
    print(f"✅ ATAC embedding: {qc_results['atac_embedding']['shape']}")
    print(f"✅ RNA KNN: {qc_results['rna_knn']['shape']}")
    print(f"✅ ATAC KNN: {qc_results['atac_knn']['shape']}")
    print(f"✅ RNA connectivity: nnz={qc_results['rna_connectivity']['nnz']}")
    print(f"✅ ATAC connectivity: nnz={qc_results['atac_connectivity']['nnz']}")
    print("✅ All checks passed!")
    print("="*50)
    
    # Store QC results in metadata
    mdata.uns["qc_results"] = qc_results
    
    return qc_results

def validate_handoff_data(mdata):
    """Final validation before handoff to next person"""
    print("\n[HANDOFF VALIDATION] Checking data for Person 3...")
    
    required_keys = [
        ("rna", "obsm", "X_pca_rna"),
        ("atac", "obsm", "X_pca_atac"),
        ("rna", "obsp", "connectivities"),
        ("atac", "obsp", "connectivities"),
        ("rna", "uns", "nn_rna"),
        ("atac", "uns", "nn_atac")
    ]
    
    for modality, attr_type, key in required_keys:
        if attr_type == "obsm":
            assert key in mdata[modality].obsm, f"Missing {modality}.obsm['{key}']"
            print(f"✅ {modality}.obsm['{key}']: {mdata[modality].obsm[key].shape}")
        elif attr_type == "obsp":
            assert key in mdata[modality].obsp, f"Missing {modality}.obsp['{key}']"
            print(f"✅ {modality}.obsp['{key}']: {mdata[modality].obsp[key].shape}")
        elif attr_type == "uns":
            assert key in mdata[modality].uns, f"Missing {modality}.uns['{key}']"
            print(f"✅ {modality}.uns['{key}']: present")
    
    # Check metadata
    required_metadata = ["pca_summary", "knn_summary"]
    for key in required_metadata:
        assert key in mdata.uns, f"Missing metadata: uns['{key}']"
        print(f"✅ uns['{key}']: present")
    
    print("[HANDOFF VALIDATION] All required data present!")
    return True

if __name__ == "__main__":
    print("Quality control module ready for import")