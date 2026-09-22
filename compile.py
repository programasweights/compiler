#!/usr/bin/env python3
"""Compile a function description into a .paw program on one CUDA GPU."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn


REFERENCE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REFERENCE_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
INTERPRETER_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
INPUT_PLACEHOLDER = "{INPUT_PLACEHOLDER}"
PREFIX_TOKEN_START = 151669
TOKEN_BUDGET = 5120
MAX_PSEUDO_TOKENS = 1024

COMPILERS = {
    "standard": {
        "checkpoint": "programasweights/paw-4b-qwen3-0.6b",
        "revision": "20260407",
        "interpreter": "Qwen/Qwen3-0.6B",
        "layers": 28,
        "architecture": "qwen3",
        "runtime_id": "qwen3-0.6b-q6_k",
        "base_repo": "programasweights/Qwen3-0.6B-GGUF-Q6_K",
        "base_file": "qwen3-0.6b-q6_k.gguf",
        "base_size": 622733120,
        "base_sha256": "9a16ed5cacba959e63b62e2b6840c3eca2b51c3c3e51d31367ef8e4aafeae33c",
        "modules": {
            "q_proj": ("attn_q", "model.layers.{}.self_attn.q_proj"),
            "k_proj": ("attn_k", "model.layers.{}.self_attn.k_proj"),
            "v_proj": ("attn_v", "model.layers.{}.self_attn.v_proj"),
            "o_proj": ("attn_output", "model.layers.{}.self_attn.o_proj"),
            "gate_proj": ("ffn_gate", "model.layers.{}.mlp.gate_proj"),
            "up_proj": ("ffn_up", "model.layers.{}.mlp.up_proj"),
            "down_proj": ("ffn_down", "model.layers.{}.mlp.down_proj"),
        },
    },
    "compact": {
        "checkpoint": "programasweights/paw-4b-gpt2",
        "revision": "20260406",
        "interpreter": "gpt2",
        "layers": 12,
        "architecture": "gpt2",
        "runtime_id": "gpt2-q8_0",
        "base_repo": "programasweights/GPT2-GGUF-Q8_0",
        "base_file": "gpt2-q8_0.gguf",
        "base_size": 139804832,
        "base_sha256": "0aa260efb2cce9def922e0546b88ad731cf1a68554db73fa2d4a0949cfa958c5",
        "modules": {
            "c_attn": ("attn_qkv", "transformer.h.{}.attn.c_attn"),
            "attn_c_proj": ("attn_output", "transformer.h.{}.attn.c_proj"),
            "c_fc": ("ffn_up", "transformer.h.{}.mlp.c_fc"),
            "mlp_c_proj": ("ffn_down", "transformer.h.{}.mlp.c_proj"),
        },
    },
}


def compiler_prompt_examples(spec: str) -> str:
    """The reference-model prompt used by the released compilers."""
    return f"""You are PAW-Compiler. Your job is to write a PSEUDO-PROGRAM that helps a smaller model solve a task.

CRITICAL: The interpreter will NOT see the original SPEC. Your pseudo-program is the ONLY instruction it gets.

Your pseudo-program should contain:
1. A clear, concise description of the task (what to do, edge cases, output format)
2. 3-6 example input/output pairs that demonstrate the task

Format (MUST follow exactly):
[PSEUDO_PROGRAM]
Task: <one paragraph describing what to do, including edge cases and output format>

Examples:
Input: <example input 1>
Output: <example output 1>

Input: <example input 2>
Output: <example output 2>

... (more examples as needed)
[END_PSEUDO_PROGRAM]

Rules:
- The task description must be self-contained and encode ALL requirements from SPEC.
- Examples should cover typical cases AND edge cases mentioned in SPEC.
- Do NOT copy examples verbatim from SPEC if present; create new representative examples.
- Keep total length under 250 tokens.
- Always include the closing marker [END_PSEUDO_PROGRAM].

Now write a pseudo-program for this specification:
[SPEC]
{spec}
[END_SPEC]
""".strip()


def render_chat(tokenizer, text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )


def pseudo_token_budget(prompt_tokens: int) -> int:
    remaining = min(MAX_PSEUDO_TOKENS, TOKEN_BUDGET - prompt_tokens)
    if remaining < 64:
        raise ValueError("The function description is too long; shorten it and try again.")
    return remaining


def release_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def generate_pseudo_program(spec: str, device: torch.device, dtype: torch.dtype) -> str:
    """Generate reference instructions, then free the reference model's GPU memory."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(REFERENCE_MODEL, revision=REFERENCE_REVISION)
    rendered = render_chat(tokenizer, compiler_prompt_examples(spec))
    inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]
    max_new_tokens = pseudo_token_budget(input_length)
    model = AutoModelForCausalLM.from_pretrained(
        REFERENCE_MODEL,
        revision=REFERENCE_REVISION,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device).eval()
    try:
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(output[0, input_length:], skip_special_tokens=True).strip()
    finally:
        del model
        release_gpu()


def build_compiler_input(tokenizer, spec: str, pseudo_program: str, prefix_steps: int) -> list[int]:
    """Use the checkpoint's trained prefix tokens without adding or resizing tokens."""
    prefix_tokens = [f"<prefix_{i}>" for i in range(1, prefix_steps + 1)]
    expected_ids = list(range(PREFIX_TOKEN_START, PREFIX_TOKEN_START + prefix_steps))
    if tokenizer.convert_tokens_to_ids(prefix_tokens) != expected_ids:
        raise ValueError("The checkpoint tokenizer is missing its trained prefix tokens.")
    prompt = f"[SPEC]\n{spec}\n[END_SPEC]\n\n[PSEUDO_PROGRAM]"
    text = render_chat(tokenizer, prompt) + pseudo_program + tokenizer.eos_token + "".join(prefix_tokens)
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    if input_ids[-prefix_steps:] != expected_ids:
        raise ValueError("The checkpoint tokenizer did not preserve the trained prefix tokens.")
    if len(input_ids) > TOKEN_BUDGET:
        raise ValueError("The compiled description exceeds the token budget; shorten the description.")
    return input_ids


def pool_prefix(hidden_states: tuple, num_student_layers: int, prefix_steps: int) -> torch.Tensor:
    """Align interpreter layers to compiler depth, then average layers and prefix positions."""
    num_teacher_layers = len(hidden_states) - 1
    indices = [
        max(0, min(num_teacher_layers - 1, ((s + 1) * num_teacher_layers // num_student_layers) - 1)) + 1
        for s in range(num_student_layers)
    ]
    # HF hidden_states[-1] is the final normalized layer, as used during training.
    prefixes = torch.stack([hidden_states[i][:, -prefix_steps:].float() for i in indices])
    return prefixes.mean(dim=0).mean(dim=1)


@torch.inference_mode()
def encode_program(checkpoint: Path, spec: str, pseudo_program: str, metadata: dict,
                   num_student_layers: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    from transformers import AutoModel, AutoTokenizer

    model_path = checkpoint / "compiler"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prefix_steps = metadata["prefix_steps"]
    token_ids = build_compiler_input(tokenizer, spec, pseudo_program, prefix_steps)
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation="sdpa",
    ).to(device).eval()
    try:
        if max(token_ids) >= model.get_input_embeddings().num_embeddings:
            raise ValueError("The compiler checkpoint lacks its trained prefix embeddings.")
        outputs = model(
            input_ids=torch.tensor([token_ids], device=device),
            output_hidden_states=True,
            use_cache=False,
        )
        pooled = pool_prefix(outputs.hidden_states, num_student_layers, prefix_steps).cpu()
        del outputs
        return pooled
    finally:
        del model
        release_gpu()


class LoraMapper(nn.Module):
    """Map the pooled compiler state to per-layer LoRA matrices using learned bases."""

    def __init__(self, state: dict[str, torch.Tensor]):
        super().__init__()
        self.module_names = sorted(key[len("A_bases_"):] for key in state if key.startswith("A_bases_"))
        if not self.module_names:
            raise ValueError("The mapper checkpoint contains no LoRA bases.")
        first = state[f"A_bases_{self.module_names[0]}"]
        self.num_bases, self.rank, _ = first.shape
        hidden_size = state["trunk.0.weight"].shape[0]
        head_size = state["coeff_head.weight"].shape[0]
        divisor = len(self.module_names) * self.num_bases * 2
        if head_size % divisor:
            raise ValueError("The mapper coefficient dimensions are inconsistent.")
        self.num_layers = head_size // divisor
        self.trunk = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU())
        self.coeff_head = nn.Linear(hidden_size, head_size)
        for name in self.module_names:
            a, b = state[f"A_bases_{name}"], state[f"B_bases_{name}"]
            if (a.ndim != 3 or b.ndim != 3 or a.shape[:2] != (self.num_bases, self.rank)
                    or (b.shape[0], b.shape[2]) != (self.num_bases, self.rank)):
                raise ValueError(f"The mapper basis dimensions for {name} are inconsistent.")
            self.register_buffer(f"A_bases_{name}", a.float())
            self.register_buffer(f"B_bases_{name}", b.float())
        self.load_state_dict(state, strict=True)
        self.eval()

    @torch.inference_mode()
    def forward(self, pooled: torch.Tensor) -> dict:
        coeffs = self.coeff_head(self.trunk(pooled.float())).reshape(
            self.num_layers, len(self.module_names), self.num_bases, 2,
        )
        params = {}
        for mi, name in enumerate(self.module_names):
            a_bases = getattr(self, f"A_bases_{name}")
            b_bases = getattr(self, f"B_bases_{name}")
            for layer in range(self.num_layers):
                a = torch.einsum("n,nrd->rd", coeffs[layer, mi, :, 0], a_bases)
                b = torch.einsum("n,ndr->dr", coeffs[layer, mi, :, 1], b_bases)
                params[layer, name] = (a.to(torch.bfloat16).cpu(), b.to(torch.bfloat16).cpu())
        return params


def render_interpreter_prompt(pseudo_program: str, config: dict) -> str:
    raw = f"{pseudo_program.strip()}\n\n[INPUT]\n{INPUT_PLACEHOLDER}\n[END_INPUT]"
    if raw.count(INPUT_PLACEHOLDER) != 1:
        raise ValueError(
            "The generated instructions contain the reserved {INPUT_PLACEHOLDER} marker. "
            "Rephrase the function description and compile again."
        )
    if config["interpreter"] == "gpt2":
        return raw
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["interpreter"], revision=INTERPRETER_REVISION)
    return render_chat(tokenizer, raw)


def write_gguf(params: dict, output: Path, config: dict, lora_alpha: float) -> None:
    import gguf
    from gguf.quants import quantize

    writer = gguf.GGUFWriter(str(output), arch=config["architecture"])
    try:
        writer.add_string("general.type", "adapter")
        writer.add_string("adapter.type", "lora")
        writer.add_float32("adapter.lora.alpha", lora_alpha)
        quantization = gguf.GGMLQuantizationType.Q4_0
        for (layer, module), (a, b) in sorted(params.items()):
            name = f"blk.{layer}.{config['modules'][module][0]}.weight"
            for suffix, tensor in (("lora_a", a), ("lora_b", b)):
                writer.add_tensor(
                    f"{name}.{suffix}", quantize(tensor.float().numpy(), quantization),
                    raw_dtype=quantization,
                )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()


def runtime_manifest(config: dict) -> dict:
    compact = config["interpreter"] == "gpt2"
    base_model = {
        "provider": "huggingface", "repo": config["base_repo"], "file": config["base_file"],
        "url": f"https://huggingface.co/{config['base_repo']}/resolve/main/{config['base_file']}",
        "size_bytes": config["base_size"], "sha256": config["base_sha256"],
    }
    return {
        "runtime_id": config["runtime_id"], "manifest_version": 1,
        "interpreter": config["interpreter"], "adapter_format": "gguf_lora",
        "prompt_template": {"format": "rendered_text", "placeholder": INPUT_PLACEHOLDER},
        "program_assets": {"adapter_filename": "adapter.gguf", "prefix_cache_required": compact,
                           "prefix_cache_filename": "prefix_cache.bin" if compact else None,
                           "prefix_tokens_filename": "prefix_tokens.json" if compact else None},
        "local_sdk": {"supported": True, "base_model": base_model, "n_ctx": 2048},
        "js_sdk": {"supported": False, "base_model": None, "prefix_cache_supported": False},
        "capabilities": {"python_local": True, "js_browser": False},
    }


def export_adapter(params: dict, output: Path, config: dict, spec: str,
                   prompt_template: str, rank: int, alpha: float) -> None:
    """Write the three files accepted by the ProgramAsWeights upload API."""
    from safetensors.torch import save_file

    output.mkdir(parents=True, exist_ok=True)
    weights, target_modules = {}, set()
    for (layer, module), (a, b) in sorted(params.items()):
        module_path = config["modules"][module][1].format(layer)
        target_modules.add(module_path.rsplit(".", 1)[-1])
        prefix = f"base_model.model.{module_path}"
        weights[f"{prefix}.lora_A.weight"] = a.contiguous()
        weights[f"{prefix}.lora_B.weight"] = b.contiguous()
    save_file(weights, str(output / "adapter_model.safetensors"))
    adapter_config = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "base_model_name_or_path": config["interpreter"],
        "r": rank, "lora_alpha": alpha, "target_modules": sorted(target_modules),
        "bias": "none", "fan_in_fan_out": config["interpreter"] == "gpt2",
    }
    manifest = {
        "schema_version": 1, "spec": spec, "runtime_id": config["runtime_id"],
        "producer": "programasweights-compiler", "prompt_template": prompt_template,
    }
    (output / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def bundle_program(output: Path, adapter: Path, config: dict, spec: str, pseudo_program: str,
                   prompt_template: str, metadata: dict, checkpoint_name: str) -> None:
    runtime = runtime_manifest(config)
    fingerprint = hashlib.sha256(adapter.read_bytes()).hexdigest()
    program_id = hashlib.sha256((fingerprint + prompt_template).encode()).hexdigest()[:20]
    meta = {
        "version": 4, "runtime_manifest_version": 1, "program_id": program_id,
        "spec": spec, "compiler_snapshot": checkpoint_name,
        "compiler_kind": "mapper_lora", "interpreter": config["interpreter"],
        "runtime_id": config["runtime_id"], "pseudo_program_strategy": "generated",
        "lora_rank": metadata["lora_rank"], "lora_alpha": metadata["lora_alpha"],
        "prefix_steps": metadata["prefix_steps"], "runtime": runtime,
        "adapter": {"format": "gguf_lora", "lora_rank": metadata["lora_rank"],
                    "lora_alpha": metadata["lora_alpha"], "target_modules": sorted(config["modules"])},
        "program_assets": runtime["program_assets"], "prompt_template": runtime["prompt_template"],
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with zipfile.ZipFile(output, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("meta.json", json.dumps(meta, indent=2))
        archive.writestr("pseudo_program.txt", pseudo_program)
        archive.writestr("prompt_template.txt", prompt_template)
        archive.write(adapter, "adapter.gguf")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--spec", help="English description of the function")
    source.add_argument("--spec-file", type=Path, help="Read the description from a UTF-8 text file")
    parser.add_argument("--compiler", choices=COMPILERS, default="standard")
    parser.add_argument("--output", type=Path, default=Path("program.paw"))
    parser.add_argument("--export-adapter", type=Path, help="Also export a PEFT adapter directory for uploading")
    parser.add_argument("--checkpoint", type=Path, help="Use a downloaded compiler checkpoint directory")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    spec = args.spec if args.spec is not None else args.spec_file.read_text(encoding="utf-8")
    if not spec.strip():
        raise ValueError("Provide a nonempty function description.")
    if not torch.cuda.is_available():
        raise RuntimeError("Compilation requires a CUDA GPU. Select a GPU with CUDA_VISIBLE_DEVICES.")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Compilation requires a CUDA GPU with bfloat16 support (Ampere or newer).")
    config = COMPILERS[args.compiler]
    checkpoint = args.checkpoint
    checkpoint_name = "local"
    if checkpoint is None:
        from huggingface_hub import snapshot_download

        checkpoint = Path(snapshot_download(
            config["checkpoint"], revision=config["revision"],
            allow_patterns=["meta.json", "lora_mapper.pt", "compiler/*"],
        ))
        checkpoint_name = f"{config['checkpoint']}@{config['revision']}"
    metadata = json.loads((checkpoint / "meta.json").read_text(encoding="utf-8"))
    if metadata["interpreter_model"] != config["interpreter"]:
        raise ValueError("The checkpoint does not match the selected compiler.")
    state = torch.load(checkpoint / "lora_mapper.pt", map_location="cpu", weights_only=True)
    mapper = LoraMapper(state)
    del state
    if mapper.num_layers != config["layers"] or set(mapper.module_names) != set(config["modules"]):
        raise ValueError("The mapper does not match the selected interpreter.")
    if mapper.rank != metadata["lora_rank"] or metadata["prefix_steps"] != 64:
        raise ValueError("The checkpoint metadata does not match the released compiler architecture.")
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    print("Generating reference instructions…", flush=True)
    pseudo_program = generate_pseudo_program(spec, device, dtype)
    template = render_interpreter_prompt(pseudo_program, config)
    print("Compiling the program…", flush=True)
    pooled = encode_program(checkpoint, spec, pseudo_program, metadata, mapper.num_layers, device, dtype)
    params = mapper(pooled)
    del mapper
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="paw-compiler-") as temp:
        adapter = Path(temp) / "adapter.gguf"
        write_gguf(params, adapter, config, metadata["lora_alpha"])
        bundle_program(args.output, adapter, config, spec, pseudo_program, template, metadata, checkpoint_name)
    if args.export_adapter:
        export_adapter(params, args.export_adapter, config, spec, template, metadata["lora_rank"],
                       metadata["lora_alpha"])
        print(f"Upload files: {args.export_adapter}")
    elapsed = time.perf_counter() - started
    peak_gib = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    print(f"Compiled {args.output} in {elapsed:.1f} s (peak GPU memory: {peak_gib:.1f} GiB).")


if __name__ == "__main__":
    main()
