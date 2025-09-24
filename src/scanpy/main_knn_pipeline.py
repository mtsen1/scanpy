"""

Main Pipeline for Person 2: PCA/kNN Analysis
Orchestrates the complete workflow from data loading to handoff

"""

import sys
import os
from pathlib import Path

# Import all modules
from gpu_setup import init_gpu, SEED
from data_loader import load_and_validate_data, validate_data_structure
from dimensionality_reduction import compute_embeddings_for_mdata
from knn_graph import build_knn_graphs_for_mdata, validate_knn_results
from affinity_matrix import build_affinity_matrices_for_mdata, validate_affinity_matrix
from quality_control import run_full_qc, validate_handoff_data

def create_handoff_metadata(mdata, params_rna, params_atac):
    """Create comprehensive handoff metadata for Person 3"""
    handoff_data = {
        "outputs": {
            "rna_embedding": "X_pca_rna",      # (n_cells, 50) CuPy array
            "atac_embedding": "X_pca_atac",    # (n_cells, 50) CuPy array  
            "rna_connectivity": "connectivities",     # (n_cells, n_cells) CSR matrix, row-normalized (SWITCHED NAMING)
            "atac_connectivity": "connectivities"    # (n_cells, n_cells) CSR matrix, row-normalized (SWITCHED NAMING)
        },
        "parameters": {
            "n_components": 50,
            "k": 15,
            "seed": SEED,
            "metrics": {"rna": "euclidean", "atac": "euclidean"}
        },
        "methods_used": {
            "rna": params_rna["method"],
            "atac": params_atac["method"]
        },
        "data_shapes": {
            "rna_embedding": list(mdata["rna"].obsm["X_pca_rna"].shape),  # Convert to list
            "atac_embedding": list(mdata["atac"].obsm["X_pca_atac"].shape),
            "rna_connectivity": list(mdata["rna"].obsp["connectivities"].shape),
            "atac_connectivity": list(mdata["atac"].obsp["connectivities"].shape)
        },
        "next_steps": [
            "Load pbmc_person2_output.h5mu",
            "Compute per-cell modality weights", 
            "Fuse RNA and ATAC connectivity matrices",
            "Create final WNN graph for downstream analysis"
        ]
    }
    
    mdata.uns["person2_handoff"] = handoff_data
    return handoff_data

def run_person2_pipeline(input_file=None, mdata=None, output_file=None, n_components=50, k=15):
    """
    Complete part 2 pipeline: PCA/kNN analysis
    
    Parameters:
    -----------
    input_file : str
        Path to input .h5 file (10X format)
    output_file : str, optional
        Path to output .h5mu file (default: auto-generated)
    n_components : int, default=50
        Number of principal components
    k : int, default=15
        Number of nearest neighbors
    
    Returns:
    --------
    mdata : MuData
        Processed multimodal data with embeddings and connectivity
    """
    
    print("\n" + "="*60)
    print("PART 2 PIPELINE: PCA + kNN ANALYSIS")
    print("="*60)
    print(f"Input: {input_file}")
    print(f"Parameters: n_components={n_components}, k={k}")
    print("="*60)
    
    # Phase 1: GPU Setup
    print("\n[PHASE 1] GPU Environment Setup")
    print("-" * 30)
    env = init_gpu(seed=SEED)
    
    # Phase 2: Data Loading
    print("\n[PHASE 2] Data Loading & Validation")
    print("-" * 30)
    if mdata is None:
        if input_file is None:
            raise ValueError("Must provide either input_file or mdata")
        mdata = load_and_validate_data(input_file)
    else:
        print("Using in-memory MuData object for processing")

    
    # Phase 3: Dimensionality Reduction
    print("\n[PHASE 3] Dimensionality Reduction")
    print("-" * 30)
    Z_rna, Z_atac, params_rna, params_atac = compute_embeddings_for_mdata(
        mdata, n_components=n_components
    )
    
    # Phase 4: KNN Graph Construction
    print("\n[PHASE 4] KNN Graph Construction")
    print("-" * 30)
    knn_results = build_knn_graphs_for_mdata(mdata, k=k, metric="euclidean")
    
    # Phase 5: Affinity Matrix Construction
    print("\n[PHASE 5] Affinity Matrix Construction")
    print("-" * 30)
    W_rna, W_atac = build_affinity_matrices_for_mdata(mdata)
    
    # Phase 6: Quality Control
    print("\n[PHASE 6] Quality Control")
    print("-" * 30)
    qc_results = run_full_qc(mdata, n_components=n_components, k=k)
    
    # Phase 7: Handoff Preparation
    print("\n[PHASE 7] Handoff Preparation")
    print("-" * 30)
    handoff_data = create_handoff_metadata(mdata, params_rna, params_atac)
    validate_handoff_data(mdata)
    
    # Phase 8: Save Results
    print("\n[PHASE 8] Saving Results")
    print("-" * 30)
    if output_file is None:
        output_file = f"pbmc_person2_output.h5mu"
    
    # Ensure directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    mdata.write(output_file)
    print(f"✅ Saved to: {output_file}")
    
    # Final Summary
    print("\n" + "="*60)
    print("PART 2 PIPELINE COMPLETE")
    print("="*60)
    print("Handoff to PART 3:")
    print(f"- RNA embedding: {handoff_data['data_shapes']['rna_embedding']}")
    print(f"- ATAC embedding: {handoff_data['data_shapes']['atac_embedding']}")
    print(f"- RNA connectivity: {handoff_data['data_shapes']['rna_connectivity']}")
    print(f"- ATAC connectivity: {handoff_data['data_shapes']['atac_connectivity']}")
    print(f"\nNext: PART 3 will compute per-cell weights and fuse matrices into WNN graph.")
    print("="*60)
    
    return mdata

def main():
    """Command line interface"""
    if len(sys.argv) < 2:
        print("Usage: python main_pipeline.py <input_file> [output_file] [n_components] [k]")
        print("Example: python main_pipeline.py pbmc_data.h5 pbmc_output.h5mu 50 15")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    n_components = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 15
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    try:
        mdata = run_person2_pipeline(
            input_file=input_file,
            output_file=output_file,
            n_components=n_components,
            k=k
        )
        print("✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()