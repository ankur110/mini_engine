# layers/ops.py
# Purpose: JIT-build + load the stable-ABI CUDA extension once; thin wrappers for the engine.
# Ops appear at torch.ops.mini_engine.<op_name> after loading.
# Importing this compiles CUDA → import only on the GPU machine (Modal), never on the Mac.
# Needs PyTorch >= 2.10 (stable Tensor + launch-check macro).

# ══ TODO 1: imports ═══════════════════════════════════════════════
# Hint: os, torch, and `load` from torch.utils.cpp_extension

# ══ TODO 2: path to csrc/ ═════════════════════════════════════════
# Hint: <repo>/layers/ops.py → <repo>/csrc
#   os.path.abspath(__file__) → dirname twice → join "csrc"   (don't rely on the cwd)

# ══ TODO 3: target GPU arch ═══════════════════════════════════════
# Hint: os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")  — before load()

# ══ TODO 4: stable-ABI flag ═══════════════════════════════════════
# Hint: one string constant, e.g. a module-level name for "-DTORCH_TARGET_VERSION=0x020a000000000000"
#   (2.10 minimum; value straight from the tutorial). Goes into BOTH flag lists below,
#   because ops.cpp (host compiler) and kv_cache.cu (nvcc) both include stable headers.

# ══ TODO 5: build + load (module level, once) ═════════════════════
# Hint: load(...) keyword args:
#   name="mini_engine_ops"                         ← .so name, not the op namespace
#   sources=[<csrc>/ops.cpp, <csrc>/kv_cache.cu]   ← C2/C3 append
#   extra_cflags=["-O3", <stable flag>]
#   extra_cuda_cflags=["-O3", "-lineinfo", <stable flag>]
#   is_python_module=False                         ← returns None; ops register on load
#   verbose=True
# Don't assign the result.

# ══ TODO 6: wrapper ═══════════════════════════════════════════════
# Hint: write the def line yourself:
#   name: store_kvcache; args in schema order: key, value, k_cache, v_cache, slot_mapping
#   body: torch.ops.mini_engine.store_kvcache(key, value, k_cache, v_cache, slot_mapping)
#   returns nothing; no Python-side checks (C++ STD_TORCH_CHECKs raise clear errors)