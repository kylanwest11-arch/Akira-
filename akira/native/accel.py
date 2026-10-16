"""Where training runs: an NVIDIA GPU when there is one, otherwise the CPU.

On CPU, matrix maths runs in bfloat16 (fast on CPUs with AMX/AVX-512 BF16).
On GPU, bfloat16 where the card supports it (A100, L4, RTX 30xx and newer);
older cards such as the T4 run in full precision, which is still many times
faster than a CPU.
"""
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast():
    if DEVICE.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported())
    return torch.autocast("cpu", dtype=torch.bfloat16)


def describe():
    if DEVICE.type == "cuda":
        return f"GPU: {torch.cuda.get_device_name(0)}"
    return f"CPU: {torch.get_num_threads()} threads"
