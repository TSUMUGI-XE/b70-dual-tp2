# PoC shim: torch-xpu 2.11 on Arc Pro B70 (0xe223) lacks free-memory query
# (_xpu_getMemoryInfo raises). vLLM 0.21 calls torch.xpu.mem_get_info during
# MemorySnapshot at init_device and crashes. total_memory IS available via
# device props, so synthesize (free, total) using torch's own reserved stats.
# Auto-imported by every interpreter (incl. spawn subprocesses) via PYTHONPATH.
try:
    import torch

    if hasattr(torch, "xpu"):
        _orig = torch.xpu.mem_get_info

        def _safe_mem_get_info(device=None):
            try:
                return _orig(device)
            except Exception:
                dev = device if device is not None else torch.xpu.current_device()
                total = int(torch.xpu.get_device_properties(dev).total_memory)
                try:
                    reserved = int(torch.xpu.memory_reserved(dev))
                except Exception:
                    reserved = 0
                free = max(total - reserved, 0)
                return (free, total)

        torch.xpu.mem_get_info = _safe_mem_get_info
        # vllm calls current_platform.mem_get_info -> torch.xpu.mem_get_info
        try:
            import torch.xpu as _x
            _x.mem_get_info = _safe_mem_get_info
        except Exception:
            pass
except Exception:
    pass
