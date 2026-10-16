"""Akira's chat format, shared by training and inference.

    <|system|>
    ...instructions...<|end|>
    <|user|>
    What's the capital of France?<|end|>
    <|assistant|>
    Paris.<|end|>
    <|endoftext|>
"""
EOT = "<|endoftext|>"
END = "<|end|>"
ROLES = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>", "tool": "<|tool|>"}
SPECIAL_TOKENS = [EOT, END, *ROLES.values()]


def format_chat(messages, add_generation_prompt=False):
    parts = []
    for m in messages:
        role = m["role"] if m["role"] in ROLES else "user"
        parts.append(f"{ROLES[role]}\n{m['content'].strip()}{END}\n")
    if add_generation_prompt:
        parts.append(f"{ROLES['assistant']}\n")
    return "".join(parts)
