
# gpu_setup.py - Fixed version
import os
import ctypes
import subprocess
import sys

def install_requirements():
    """Install packages if needed"""
    # DON'T import cuML here - just check if basic packages exist
    try:
        import numpy
        print("✅ Basic packages available")
    except ImportError:
        print("📦 Installing missing packages...")
        
        packages = [
            "numpy", "scipy", "pandas", "matplotlib", "seaborn", "psutil",
            "scanpy", "muon", "anndata", "h5py", "tables", "scikit-learn"
        ]
        
        for package in packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])
            except subprocess.CalledProcessError:
                print(f"⚠️ Failed to install {package}")
        
        # GPU packages
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "cupy-cuda12x", "cuml-cu12"])
        except subprocess.CalledProcessError:
            print("⚠️ GPU packages failed")

def fix_cuda_libraries():
    """Fix CUDA library loading issues"""
    base_path = '/home/shadeform/.local/lib/python3.10/site-packages/nvidia'
    
    cuda_libs = [
        ('cuda_runtime', 'libcudart.so.12'),
        ('cuda_nvrtc', 'libnvrtc.so.12'),
        ('cuda_nvrtc', 'libnvrtc-builtins.so.12'),
        ('nvjitlink', 'libnvjitlink.so.12'),
        ('cublas', 'libcublas.so.12'),
        ('cufft', 'libcufft.so.12'),
        ('cusolver', 'libcusolver.so.12'),
        ('cusparse', 'libcusparse.so.12'),
        ('curand', 'libcurand.so.12')
    ]
    
    loaded_libs = []
    failed_libs = []
    
    for lib_dir, lib_name in cuda_libs:
        lib_path = os.path.join(base_path, lib_dir, 'lib', lib_name)
        try:
            if os.path.exists(lib_path):
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                loaded_libs.append(lib_name)
            else:
                failed_libs.append(f"{lib_name} (not found at {lib_path})")
        except Exception as e:
            failed_libs.append(f"{lib_name} (error: {e})")
    
    # Set LD_LIBRARY_PATH
    lib_paths = []
    for lib_dir, _ in cuda_libs:
        lib_path = os.path.join(base_path, lib_dir, 'lib')
        if os.path.exists(lib_path):
            lib_paths.append(lib_path)
    
    if lib_paths:
        current_path = os.environ.get('LD_LIBRARY_PATH', '')
        new_path = ':'.join(lib_paths)
        os.environ['LD_LIBRARY_PATH'] = f"{new_path}:{current_path}" if current_path else new_path
    
    print(f"✅ Loaded {len(loaded_libs)} libraries: {', '.join(loaded_libs)}")
    if failed_libs:
        print(f"⚠️ Failed to load {len(failed_libs)} libraries: {', '.join(failed_libs)}")
    
    return len(loaded_libs) > 0

def init_gpu(seed=42, use_rmm=False):
    """Initialize GPU environment with proper order"""
    print("📦 Checking basic packages...")
    install_requirements()
    
    print("🔧 Fixing CUDA libraries...")
    fix_cuda_libraries()
    
    print("📥 Importing GPU libraries...")
    # Import GPU libraries AFTER CUDA fix
    try:
        import cupy as cp
        from cuml.common import set_global_output_type
        
        test_array = cp.array([1, 2, 3])
        print(f"✅ CuPy working: {test_array}")
        
        cp.random.seed(seed)
        set_global_output_type("cupy")
        
        env_config = {"seed": seed, "rmm_enabled": use_rmm}
        print(f"[GPU READY] {env_config}")
        
        return env_config
        
    except Exception as e:
        print(f"❌ GPU initialization failed: {e}")
        raise

# Constants
SEED = 42
EPS = 1e-12

if __name__ == "__main__":
    env = init_gpu(seed=SEED)
    print("GPU setup completed successfully!")
