"""Where training runs: an NVIDIA GPU, an Apple-silicon GPU (M1/M2/M3/M4), or the CPU.

- NVIDIA: bfloat16 where the card supports it (A100, L4, RTX 30xx and newer); older cards
  such as the T4 run in full precision, still many times faster than a CPU.
- Apple silicon: the Mac's built-in GPU ("mps") in full precision. A few operations the Mac
  GPU lacks (such as the speech brain's CTC loss) quietly run on the CPU instead.
- CPU: bfloat16 matrix maths on CPUs built for it (AMX or AVX-512 BF16); full precision on
  others, where bfloat16 is emulated and many times slower.
"""
import contextlib
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # let unsupported Mac GPU ops fall back to the CPU

import torch  # noqa: E402


def _pick():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _pick()


def _cpu_has_bf16():
    try:
        flags = open("/proc/cpuinfo").read()
    except OSError:
        return False
    return "amx_bf16" in flags or "avx512_bf16" in flags


CPU_BF16 = DEVICE.type == "cpu" and _cpu_has_bf16()


def autocast():
    if DEVICE.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported())
    if DEVICE.type == "mps":
        return contextlib.nullcontext()      # full precision: the most reliable choice on Apple GPUs
    return torch.autocast("cpu", dtype=torch.bfloat16, enabled=CPU_BF16)


def describe():
    if DEVICE.type == "cuda":
        return f"GPU: {torch.cuda.get_device_name(0)}"
    if DEVICE.type == "mps":
        return "GPU: Apple silicon (mps)"
    return f"CPU: {torch.get_num_threads()} threads, {'bfloat16' if CPU_BF16 else 'full precision'}"
