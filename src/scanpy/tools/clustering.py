# part4_clustering.py
import cupy as cp
import cudf
import cugraph
from scipy.sparse import csr_matrix

HAS_CUGRAPH = True
try:
    import cudf
    import cupy as cp
    import cugraph
except ImportError:
    HAS_CUGRAPH = False


def csr_to_cugraph(csr_mat):
    """
    Convert a SciPy CSR adjacency matrix to a cuGraph Graph.
    """
    coo = csr_mat.tocoo()
    edges = cudf.DataFrame({
        "src": coo.row,
        "dst": coo.col,
        "weight": coo.data
    })
    G = cugraph.Graph()
    G.from_cudf_edgelist(edges, source='src', destination='dst', edge_attr='weight', renumber=False)
    return G


def run_gpu_clustering(G, method="leiden", resolution=1.0, random_state=0):
    """
    Run GPU-based community detection (Leiden or Louvain) on a cuGraph Graph.
    Returns a cuDF DataFrame with columns 'vertex' and 'partition'.
    """
    import cugraph.community as community

    if method.lower() == "leiden":
        result = community.leiden(G, resolution=resolution, random_state=random_state)
        if isinstance(result, tuple):
            result = result[0]  # some versions return tuple (df, modularity)
    elif method.lower() == "louvain":
        result = community.louvain(G)
    else:
        raise ValueError("Method must be 'leiden' or 'louvain'")
    return result


def map_clusters_to_obs(mdata, cluster_df, obs_key="wnn_leiden"):
    """
    Map GPU cluster results back to mdata.obs with proper alignment.
    """
    cluster_pd = cluster_df.to_pandas()
    cluster_pd['cell'] = mdata.obs.index[cluster_pd['vertex'].values]
    cluster_pd.set_index('cell', inplace=True)
    mdata.obs[obs_key] = cluster_pd['partition'].astype(str)


def run_part4_clustering(
    mdata,
    obs_key="wnn_leiden",
    method="leiden",
    resolution=1.0,
    random_state=0
):
    """
    Full Part 4 pipeline:
    - Convert CuPy CSR → SciPy CSR if needed
    - Add self-loops
    - Build cuGraph Graph
    - Run GPU clustering
    - Map results to mdata.obs
    Returns number of clusters detected.
    """
    if not HAS_CUGRAPH:
        print("CuGraph/CuML not available — skipping Part 4")
        return 0

    # Get fused WNN matrix
    wnn_cupy = mdata.obsp["wnn_connectivities"]

    # Convert CuPy CSR → SciPy CSR if needed
    if isinstance(wnn_cupy, cp.sparse.csr_matrix) or "cupyx" in str(type(wnn_cupy)):
        wnn_scipy = csr_matrix(wnn_cupy.get())
    else:
        wnn_scipy = wnn_cupy

    # Optional: add self-loops
    wnn_scipy.setdiag(1.0)

    # Build cuGraph
    G = csr_to_cugraph(wnn_scipy)

    # Run clustering
    clusters_df = run_gpu_clustering(G, method=method, resolution=resolution, random_state=random_state)

    # Map back to obs
    map_clusters_to_obs(mdata, clusters_df, obs_key=obs_key)

    # QC
    n_clusters = mdata.obs[obs_key].nunique()
    print(f"Part 4 complete: {n_clusters} clusters detected")
    return n_clusters
