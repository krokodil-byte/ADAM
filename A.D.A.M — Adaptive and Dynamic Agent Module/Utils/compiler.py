#!/usr/bin/env python3
"""
CUDA Compiler Utilities
Auto-compilation del kernel CUDA con caching
"""

import os
import sys
import subprocess
import hashlib
from pathlib import Path
from typing import Optional

# Import config
try:
    from core.config import RUNTIME_CONFIG
except ImportError:
    # Fallback per test standalone
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.config import RUNTIME_CONFIG


class CUDACompiler:
    """Compilatore automatico del kernel CUDA con caching intelligente"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Args:
            cache_dir: Directory cache (default da config)
        """
        self.cache_dir = cache_dir or RUNTIME_CONFIG.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_cache_path(self, kernel_path: Path) -> Path:
        """
        Genera path cache basato su hash del kernel.
        Se il kernel cambia, ricompila automaticamente.
        
        Args:
            kernel_path: Path al file .cu
            
        Returns:
            Path alla libreria compilata cached
        """
        with open(kernel_path, 'rb') as f:
            code_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        return self.cache_dir / f"libvectllm_{code_hash}.so"
    
    def find_nvcc(self) -> str:
        """
        Trova il compilatore nvcc.
        
        Cerca in:
        1. Posizioni comuni (/usr/local/cuda/bin, /usr/bin)
        2. PATH
        
        Returns:
            Path a nvcc
            
        Raises:
            RuntimeError: Se nvcc non trovato
        """
        # Try common locations
        for path in ['/usr/local/cuda/bin/nvcc', '/usr/bin/nvcc']:
            if os.path.exists(path):
                return path
        
        # Try PATH
        result = subprocess.run(['which', 'nvcc'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        
        raise RuntimeError(
            "❌ NVCC not found. Install CUDA toolkit.\n"
            "   See: https://developer.nvidia.com/cuda-downloads"
        )
    
    def detect_gpu_arch(self) -> str:
        """
        Rileva compute capability GPU usando nvidia-smi.
        
        Returns:
            Architecture string (es. "sm_86")
        """
        if RUNTIME_CONFIG.NVCC_ARCH != "auto":
            return RUNTIME_CONFIG.NVCC_ARCH
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                cap = result.stdout.strip().replace('.', '')
                arch = f"sm_{cap}"
                print(f"   Detected GPU: {arch}")
                return arch
        except Exception as e:
            print(f"⚠️  Could not detect GPU arch: {e}")
        
        # Default to Ampere (RTX 30xx, A100)
        print("⚠️  Using default arch: sm_86 (Ampere)")
        return "sm_86"
    
    def compile(self, kernel_path: Path, force: bool = False) -> Path:
        """
        Compila il kernel CUDA.
        
        Args:
            kernel_path: Path al file brain.cu
            force: Se True, ricompila anche se cached
            
        Returns:
            Path alla libreria compilata
            
        Raises:
            RuntimeError: Se compilazione fallisce
        """
        lib_path = self.get_cache_path(kernel_path)
        
        # Check cache
        if lib_path.exists() and not force:
            print(f"✓ Using cached library: {lib_path.name}")
            return lib_path
        
        print("🔨 Compiling CUDA kernel...")
        print(f"   Source: {kernel_path}")
        
        # Find tools
        nvcc = self.find_nvcc()
        arch = self.detect_gpu_arch()
        
        print(f"   NVCC: {nvcc}")
        print(f"   Arch: {arch}")
        
        # Build command
        cmd = [
            nvcc,
            str(kernel_path),
            '-o', str(lib_path)
        ] + RUNTIME_CONFIG.NVCC_FLAGS + [f'-arch={arch}']
        
        if RUNTIME_CONFIG.VERBOSE:
            print(f"   Command: {' '.join(cmd)}")
        
        # Compile
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                print(f"❌ Compilation failed:")
                print(result.stderr)
                raise RuntimeError("CUDA compilation failed")
            
            if result.stderr and RUNTIME_CONFIG.VERBOSE:
                print(f"⚠️  Compilation warnings:\n{result.stderr}")
            
            print(f"✅ Compiled successfully: {lib_path.name}")
            return lib_path
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Compilation timeout (120s)")
        except Exception as e:
            raise RuntimeError(f"Compilation error: {e}")
    
    def clean_cache(self, keep_latest: int = 3):
        """
        Pulisce vecchie librerie compilate.
        
        Args:
            keep_latest: Numero di versioni da mantenere
        """
        libs = sorted(
            self.cache_dir.glob("libvectllm_*.so"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        for lib in libs[keep_latest:]:
            print(f"🗑️  Removing old library: {lib.name}")
            lib.unlink()


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("=== CUDA Compiler Test ===\n")
    
    compiler = CUDACompiler()
    
    # Test 1: Find nvcc
    try:
        nvcc = compiler.find_nvcc()
        print(f"✅ Found nvcc: {nvcc}\n")
    except RuntimeError as e:
        print(f"❌ {e}\n")
    
    # Test 2: Detect GPU
    arch = compiler.detect_gpu_arch()
    print(f"✅ GPU arch: {arch}\n")
    
    # Test 3: Compile (if brain.cu exists)
    kernel_path = Path(__file__).parent.parent / "kernels" / "brain.cu"
    if kernel_path.exists():
        try:
            lib_path = compiler.compile(kernel_path)
            print(f"\n✅ Compilation test passed: {lib_path}")
        except Exception as e:
            print(f"\n❌ Compilation test failed: {e}")
    else:
        print(f"⚠️  Kernel not found: {kernel_path}")
    
    print("\n✅ Compiler tests complete!")
