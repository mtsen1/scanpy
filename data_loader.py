"""

Data Loading and Validation
Handles MuData loading and basic validation

"""

import numpy as np
import muon as mu
from mudata import MuData

def load_and_validate_data(file_or_mdata):
    """Load MuData and perform basic validation"""
    print(f"Loading data from: {file_or_mdata}")

    # CHANGED
    # Case 1: already a MuData object
    if isinstance(file_or_mdata, MuData):
        mdata = file_or_mdata
        print("Using existing MuData object")

    # Case 2: string path
    elif isinstance(file_or_mdata, str):
        print(f"Loading data from: {file_or_mdata}")
        if file_or_mdata.endswith(".h5mu"):
            mdata = mu.read(file_or_mdata)
            print("Loaded existing MuData file")
        else:
            mdata = mu.read_10x_h5(file_or_mdata)
            print("Loaded 10X HDF5 file")

    else:
        raise TypeError(f"Expected str or MuData, got {type(file_or_mdata)}")
    
    # Make variable names unique if needed
    if "rna" in mdata.mod:
        mdata["rna"].var_names_make_unique()
    if "atac" in mdata.mod:
        mdata["atac"].var_names_make_unique()
    
    # Convert to float32 for GPU efficiency
    if "rna" in mdata.mod:
        mdata["rna"].X = mdata["rna"].X.astype(np.float32, copy=False)
    if "atac" in mdata.mod:
        mdata["atac"].X = mdata["atac"].X.astype(np.float32, copy=False)
    
    # Alignment check
    if "rna" in mdata.mod and "atac" in mdata.mod:
        assert mdata["rna"].obs_names.equals(mdata["atac"].obs_names), "Cell alignment failed"
    
    print(f"[DATA] Cells: {mdata.n_obs}, RNA features: {mdata['rna'].n_vars}, ATAC features: {mdata['atac'].n_vars}")
    
    return mdata

def validate_data_structure(mdata):
    """Additional validation checks"""
    # Check for required modalities
    assert "rna" in mdata.mod, "RNA modality missing"
    assert "atac" in mdata.mod, "ATAC modality missing"
    
    # Check data types
    assert mdata["rna"].X.dtype == np.float32, f"RNA data not float32: {mdata['rna'].X.dtype}"
    assert mdata["atac"].X.dtype == np.float32, f"ATAC data not float32: {mdata['atac'].X.dtype}"
    
    # Check for minimum cells and features
    min_cells = 100
    min_rna_features = 1000
    min_atac_features = 1000
    
    assert mdata.n_obs >= min_cells, f"Too few cells: {mdata.n_obs} < {min_cells}"
    assert mdata["rna"].n_vars >= min_rna_features, f"Too few RNA features: {mdata['rna'].n_vars} < {min_rna_features}"
    assert mdata["atac"].n_vars >= min_atac_features, f"Too few ATAC features: {mdata['atac'].n_vars} < {min_atac_features}"
    
    print("[VALIDATION] Data structure checks passed")
    return True

if __name__ == "__main__":
    # Test data loading
    file_path = "pbmc_unsorted_10k_filtered_feature_bc_matrix.h5"
    mdata = load_and_validate_data(file_path)
    validate_data_structure(mdata)
    print("Data loading test completed!")