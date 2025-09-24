
"""

PCA and Truncated SVD Implementation
Handles dimensionality reduction for both dense and sparse data

"""

import numpy as np
import scipy.sparse as sp
from gpu_setup import SEED

def run_pca_dense(X, n_components=50):
    """cuML PCA on dense data"""
    from cuml.decomposition import PCA as GPUPCA
    import cupy as cp
    
    # Convert to GPU array
    Xg = cp.asarray(X, dtype=cp.float32, order="C")
    m, n = Xg.shape
    k = max(1, min(n_components, min(m, n) - 1))
    
    # Fit PCA
    pca = GPUPCA(n_components=k)
    Z = pca.fit_transform(Xg)
    
    params = {
        "method": "pca_dense_cuml",
        "n_components": k,
        "explained_variance_ratio_": cp.asnumpy(pca.explained_variance_ratio_) if hasattr(pca, "explained_variance_ratio_") else None
    }
    
    return Z.astype(cp.float32), params

def run_tsvd_sparse(X, n_components=50, seed=SEED):
    """TSVD for sparse data with CPU fallback"""
    try:
        # Try cuML TSVD on GPU
        from cuml.decomposition import TruncatedSVD as TSVD
        import cupy as cp
        import cupyx.scipy.sparse as cpx_sp
        
        # Convert to GPU sparse matrix
        Xcsr = X.tocsr().astype(np.float32)
        data = cp.asarray(Xcsr.data)
        indices = cp.asarray(Xcsr.indices, dtype=cp.int32)
        indptr = cp.asarray(Xcsr.indptr, dtype=cp.int32)
        Xg = cpx_sp.csr_matrix((data, indices, indptr), shape=Xcsr.shape)
        
        # Fit TSVD
        tsvd = TSVD(n_components=n_components)
        Z = tsvd.fit_transform(Xg)
        
        params = {"method": "tsvd_sparse_cuml", "n_components": n_components}
        return Z.astype(cp.float32), params
        
    except Exception as e:
        print(f"GPU TSVD failed ({e}), falling back to CPU")
        
        # CPU fallback using sklearn
        from sklearn.decomposition import TruncatedSVD as SKTSVD
        
        Xcsr = X.tocsr().astype(np.float32)
        k = max(1, min(n_components, min(Xcsr.shape) - 1))
        
        # Fit sklearn TSVD
        sk = SKTSVD(n_components=k, random_state=seed)
        Z_cpu = sk.fit_transform(Xcsr)
        Z = cp.asarray(Z_cpu, dtype=cp.float32)
        
        params = {
            "method": "tsvd_sparse_sklearn_cpu",
            "n_components": k,
            "explained_variance_ratio_": sk.explained_variance_ratio_
        }
        
        return Z, params

def compute_embedding(adata, label, n_components=50):
    """Auto-select PCA vs TSVD based on data characteristics"""
    X = adata.X
    is_sparse = sp.issparse(X)
    density = X.nnz / (X.shape[0] * X.shape[1]) if is_sparse else 1.0
    
    print(f"[EMBEDDING] {label}: sparse={is_sparse}, density={density:.4f}")
    
    # Choose method based on sparsity
    if is_sparse and density < 0.15:
        Z, params = run_tsvd_sparse(X, n_components)
    else:
        Xdense = X.toarray() if is_sparse else X
        Z, params = run_pca_dense(Xdense, n_components)
    
    # Store results
    adata.obsm[f"X_pca_{label}"] = Z
    adata.uns[f"pca_params_{label}"] = params
    
    print(f"[PCA] {label}: {params['method']}, shape={Z.shape}")
    return Z, params

def compute_embeddings_for_mdata(mdata, n_components=50):
    """Compute embeddings for both RNA and ATAC modalities"""
    print(f"Computing embeddings with {n_components} components...")
    
    # Compute embeddings
    Z_rna, params_rna = compute_embedding(mdata["rna"], "rna", n_components)
    Z_atac, params_atac = compute_embedding(mdata["atac"], "atac", n_components)
    
    # Store summary metadata
    mdata.uns["pca_summary"] = {
        "n_components": n_components,
        "methods": {"rna": params_rna["method"], "atac": params_atac["method"]},
        "seed": SEED
    }
    
    print(f"[EMBEDDINGS] Complete - RNA: {Z_rna.shape}, ATAC: {Z_atac.shape}")
    return Z_rna, Z_atac, params_rna, params_atac

if __name__ == "__main__":
    # Test with dummy data
    from data_loader import load_and_validate_data
    
    # This would need actual data file
    print("Dimensionality reduction module ready for import")
