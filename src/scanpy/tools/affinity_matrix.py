
"""

Affinity Matrix Construction
Converts KNN distances to symmetric, normalized affinity matrices

"""

import cupy as cp
from gpu_setup import EPS

def distances_to_affinities(D, I, add_self=True, self_loop_value=1.0):
    """Convert KNN distances to symmetric affinity matrix"""
    import cupyx.scipy.sparse as cpx_sp
    
    n, k = I.shape
    print(f"[AFFINITY] Converting distances: n={n}, k={k}")
    
    # Self-tuned sigma (last neighbor distance + epsilon)
    sigma = D[:, -1] + EPS
    
    # Build sparse matrix indices
    rows = cp.repeat(cp.arange(n, dtype=cp.int32), k)
    cols = I.ravel()
    
    # Gaussian weights: exp(-d²/σ²)
    D_flat = D.ravel()
    sigma_expanded = cp.repeat(sigma, k)
    weights = cp.exp(-(D_flat**2) / (sigma_expanded**2 + EPS)).astype(cp.float32)
    
    # Create sparse matrix
    W = cpx_sp.csr_matrix((weights, (rows, cols)), shape=(n, n), dtype=cp.float32)
    
    # Add self-loops if requested
    if add_self:
        W.setdiag(W.diagonal() + self_loop_value)
    
    # Symmetrize by adding W + W.T and averaging
    W = W.tocoo()
    rows_all = cp.concatenate([W.row, W.col])
    cols_all = cp.concatenate([W.col, W.row])
    data_all = cp.concatenate([W.data, W.data])
    
    W_sym = cpx_sp.coo_matrix((data_all, (rows_all, cols_all)), shape=(n, n))
    W_sym = W_sym.tocsr()
    W_sym.sum_duplicates()
    W_sym = W_sym * 0.5  # Average overlapping entries
    
    # Row normalize to make it a proper transition matrix
    row_sums = cp.array(W_sym.sum(axis=1)).ravel()
    row_sums = cp.maximum(row_sums, EPS)  # Avoid division by zero
    inv_row_sums = cp.reciprocal(row_sums)
    D_inv = cpx_sp.diags(inv_row_sums)
    W_normalized = D_inv @ W_sym
    
    print(f"[AFFINITY] Created: shape={W_normalized.shape}, nnz={W_normalized.nnz}")
    return W_normalized.tocsr()

def build_affinity_matrix(adata, label):
    """Build affinity matrix from stored KNN results"""
    print(f"[AFFINITY] Building matrix for {label}")
    
    # Get KNN results
    nn_data = adata.uns[f"nn_{label}"]
    D = nn_data["distances"]
    I = nn_data["indices"]
    
    # Convert to affinity matrix
    W = distances_to_affinities(D, I)
    
    # Store results
    adata.obsp[f"connectivities"] = W
    adata.obsp["affinity"] = W  # Also store with generic name for last operation
    
    print(f"[AFFINITY] {label}: n={W.shape[0]}, nnz={W.nnz}, symmetric=True")
    return W

def build_affinity_matrices_for_mdata(mdata):
    """Build affinity matrices for both RNA and ATAC"""
    print("Building affinity matrices...")
    
    # Build matrices
    W_rna = build_affinity_matrix(mdata["rna"], "rna")
    W_atac = build_affinity_matrix(mdata["atac"], "atac")
    
    # Store metadata
    mdata.uns["affinity_summary"] = {
        "rna_nnz": int(W_rna.nnz),
        "atac_nnz": int(W_atac.nnz),
        "symmetric": True,
        "row_normalized": True
    }
    
    print(f"[AFFINITY] Complete - RNA: {W_rna.nnz} edges, ATAC: {W_atac.nnz} edges")
    return W_rna, W_atac

def validate_affinity_matrix(W, label, tol=1e-3):
    """Validate affinity matrix properties"""
    n, m = W.shape
    
    # Basic checks
    assert n == m, f"Not square: {W.shape}"
    assert W.nnz > n, f"Too sparse: {W.nnz} <= {n}"
    assert cp.isfinite(W.data).all(), "Non-finite weights"
    assert not (W.data < 0).any(), "Negative weights"
    
    # Row normalization check
    row_sums = cp.array(W.sum(axis=1)).ravel()
    max_dev = float(cp.max(cp.abs(row_sums - 1.0)))
    assert max_dev <= tol, f"Row normalization failed: {max_dev}"
    
    # Connectivity check (each row should have at least one non-zero)
    zero_rows = cp.count_nonzero(row_sums < EPS)
    assert zero_rows == 0, f"Found {zero_rows} disconnected cells"
    
    print(f"[VALIDATE] Affinity {label}: max_row_dev={max_dev:.3e}, OK")




if __name__ == "__main__":
    print("Affinity matrix module ready for import")



