"""Check the trained mapper's tensor convention and portable program formats."""

import importlib.util
import json
from pathlib import Path
import zipfile

import pytest
import torch
from safetensors.torch import load_file


module_spec = importlib.util.spec_from_file_location("paw_compile", Path(__file__).parents[1] / "compile.py")
compiler = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(compiler)


def test_depth_ratio_pooling_uses_last_prefix_positions_and_final_normalization():
    # Six compiler layers map to four interpreter layers as [1, 3, 4, 6].
    states = tuple(torch.full((1, 5, 2), float(i)) for i in range(7))
    for state in states:
        state[:, :3] = -1000  # These non-prefix positions must never contribute.
    states[-1][:, -2:] = 60  # Final normalized output differs from intermediate states.
    actual = compiler.pool_prefix(states, num_student_layers=4, prefix_steps=2)
    torch.testing.assert_close(actual, torch.full((1, 2), 17.0))


def test_depth_ratio_pooling_repeats_compiler_layers_for_deeper_interpreter():
    states = tuple(torch.full((1, 2, 1), float(i)) for i in range(3))
    actual = compiler.pool_prefix(states, num_student_layers=4, prefix_steps=2)
    torch.testing.assert_close(actual, torch.tensor([[1.25]]))


def mapper_state():
    # Two modules in reverse insertion order exercise the trained alphabetical order.
    state = {
        "trunk.0.weight": torch.eye(2),
        "trunk.0.bias": torch.zeros(2),
        "coeff_head.weight": torch.zeros(16, 2),
        "coeff_head.bias": torch.arange(1, 17, dtype=torch.float32),
    }
    for module in ("z", "a"):
        state[f"A_bases_{module}"] = torch.tensor([[[1.0, 2.0]], [[10.0, 20.0]]])
        state[f"B_bases_{module}"] = torch.tensor([[[3.0], [4.0]], [[30.0], [40.0]]])
    return state


def test_mapper_restores_layer_module_basis_ab_order_and_outputs_bfloat16():
    mapper = compiler.LoraMapper(mapper_state())
    params = mapper(torch.zeros(1, 2))
    assert mapper.num_layers == 2
    assert mapper.rank == 1
    assert mapper.module_names == ["a", "z"]
    # Layer 0, module a: A mixes bases with 1 and 3; B with 2 and 4.
    torch.testing.assert_close(params[0, "a"][0].float(), torch.tensor([[31.0, 62.0]]))
    torch.testing.assert_close(params[0, "a"][1].float(), torch.tensor([[126.0], [168.0]]))
    # Layer 1, module z: A coefficients 13,15; B coefficients 14,16.
    torch.testing.assert_close(params[1, "z"][0].float(), torch.tensor([[163.0, 326.0]]))
    torch.testing.assert_close(params[1, "z"][1], torch.tensor([[522.0], [696.0]]).bfloat16())
    assert all(t.dtype == torch.bfloat16 for pair in params.values() for t in pair)


def test_mapper_trunk_and_head_use_the_input_and_full_precision_weights():
    state = mapper_state()
    state["coeff_head.bias"].zero_()
    state["coeff_head.weight"][:, 0] = 0.1234567
    mapper = compiler.LoraMapper(state)
    value = torch.tensor([[1.0, 0.0]])
    coefficient = torch.nn.functional.gelu(value)[0, 0] * 0.1234567
    expected_a = (coefficient * torch.tensor([[11.0, 22.0]])).bfloat16()
    torch.testing.assert_close(mapper(value)[0, "a"][0], expected_a)


@pytest.mark.parametrize("broken", ["coefficients", "basis"])
def test_mapper_rejects_inconsistent_checkpoint_dimensions(broken):
    state = mapper_state()
    if broken == "coefficients":
        state["coeff_head.weight"] = torch.zeros(15, 2)
    else:
        state["B_bases_a"] = torch.zeros(3, 2, 1)
    with pytest.raises(ValueError, match="dimensions"):
        compiler.LoraMapper(state)


@pytest.mark.parametrize("kind", ["standard", "compact"])
def test_peft_export_preserves_weights_and_uses_loadable_module_names(tmp_path, kind):
    config = compiler.COMPILERS[kind]
    params = {
        (0, module): (torch.ones(32, 64).bfloat16(), torch.full((96, 32), 2.0).bfloat16())
        for module in config["modules"]
    }
    template = "Task instructions\n" + compiler.INPUT_PLACEHOLDER
    compiler.export_adapter(params, tmp_path, config, "Classify support requests.", template, 32, 16)
    assert {p.name for p in tmp_path.iterdir()} == {
        "adapter_config.json", "adapter_model.safetensors", "manifest.json",
    }
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest == {
        "schema_version": 1, "spec": "Classify support requests.", "runtime_id": config["runtime_id"],
        "producer": "programasweights-compiler", "prompt_template": template,
    }
    adapter = json.loads((tmp_path / "adapter_config.json").read_text())
    weights = load_file(str(tmp_path / "adapter_model.safetensors"))
    for (layer, module), (a, b) in params.items():
        prefix = "base_model.model." + config["modules"][module][1].format(layer)
        torch.testing.assert_close(weights[prefix + ".lora_A.weight"], a)
        torch.testing.assert_close(weights[prefix + ".lora_B.weight"], b)
    if kind == "compact":
        assert adapter["target_modules"] == ["c_attn", "c_fc", "c_proj"]
        assert adapter["fan_in_fan_out"] is True
        assert "base_model.model.transformer.h.0.attn.c_proj.lora_A.weight" in weights
        assert "base_model.model.transformer.h.0.mlp.c_proj.lora_A.weight" in weights
    else:
        assert adapter["target_modules"] == sorted(config["modules"])
        assert adapter["fan_in_fan_out"] is False


@pytest.mark.parametrize("kind", ["standard", "compact"])
def test_paw_bundle_has_complete_local_runtime_and_only_public_metadata(tmp_path, kind):
    config = compiler.COMPILERS[kind]
    adapter = tmp_path / "adapter.gguf"
    adapter.write_bytes(b"GGUF-test-adapter")
    output = tmp_path / "result.paw"
    checkpoint_meta = {
        "lora_rank": 64, "lora_alpha": 16, "prefix_steps": 64,
        "compiler_model": "/private/training/checkpoint", "opt_step": 123,
    }
    template = "Instructions\n" + compiler.INPUT_PLACEHOLDER
    compiler.bundle_program(output, adapter, config, "A function.", "Pseudo-program.", template,
                            checkpoint_meta, "local")
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {
            "adapter.gguf", "meta.json", "pseudo_program.txt", "prompt_template.txt",
        }
        meta_text = archive.read("meta.json").decode()
        meta = json.loads(meta_text)
        assert "/private" not in meta_text
        assert "opt_step" not in meta
        assert "inference_provider_url" not in meta["runtime"]
        assert archive.read("adapter.gguf") == adapter.read_bytes()
        assert archive.read("prompt_template.txt").decode() == template
    assert meta["version"] == 4
    assert meta["runtime"]["local_sdk"]["n_ctx"] == 2048
    assert meta["runtime"]["local_sdk"]["base_model"]["sha256"] == config["base_sha256"]
    assert meta["program_assets"]["prefix_cache_required"] is (kind == "compact")
    assert meta["runtime"]["capabilities"]["js_browser"] is False


@pytest.mark.parametrize("kind", ["standard", "compact"])
def test_runtime_metadata_satisfies_python_sdk_contract(kind):
    sdk_cache = pytest.importorskip("programasweights.cache")
    if not hasattr(sdk_cache, "_normalize_runtime_manifest"):
        pytest.skip("Requires programasweights >= 0.4.10")
    config = compiler.COMPILERS[kind]
    normalized = sdk_cache._normalize_runtime_manifest(
        compiler.runtime_manifest(config), expected_runtime_id=config["runtime_id"], require_local_model=True,
    )
    assert normalized is not None
    assert normalized["program_assets"] == sdk_cache.LEGACY_RUNTIME_MANIFESTS[config["runtime_id"]]["program_assets"]


def test_gguf_contains_quantized_lora_tensors_and_scaling(tmp_path):
    gguf = pytest.importorskip("gguf")
    config = compiler.COMPILERS["standard"]
    params = {(0, "q_proj"): (torch.ones(32, 64), torch.ones(96, 32))}
    path = tmp_path / "adapter.gguf"
    compiler.write_gguf(params, path, config, 16)
    reader = gguf.GGUFReader(str(path))
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    assert set(tensors) == {"blk.0.attn_q.weight.lora_a", "blk.0.attn_q.weight.lora_b"}
    assert tuple(tensors["blk.0.attn_q.weight.lora_a"].shape) == (64, 32)
    assert tuple(tensors["blk.0.attn_q.weight.lora_b"].shape) == (32, 96)
    assert all(t.tensor_type == gguf.GGMLQuantizationType.Q4_0 for t in tensors.values())
    assert reader.get_field("adapter.lora.alpha").contents() == 16


def test_compact_interpreter_prompt_is_plain_text_and_preserves_placeholder():
    assert compiler.render_interpreter_prompt(" Pseudo-program.\n", compiler.COMPILERS["compact"]) == (
        "Pseudo-program.\n\n[INPUT]\n{INPUT_PLACEHOLDER}\n[END_INPUT]"
    )


@pytest.mark.parametrize("kind", ["standard", "compact"])
def test_interpreter_prompt_rejects_reserved_input_marker_in_generated_instructions(kind):
    with pytest.raises(ValueError, match="reserved.*INPUT_PLACEHOLDER"):
        compiler.render_interpreter_prompt("Read {INPUT_PLACEHOLDER}.", compiler.COMPILERS[kind])


class PrefixTokenizer:
    eos_token = "<eos>"

    def convert_tokens_to_ids(self, tokens):
        return list(range(compiler.PREFIX_TOKEN_START, compiler.PREFIX_TOKEN_START + len(tokens)))

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"add_generation_prompt": True, "tokenize": False, "enable_thinking": False}
        return "CHAT:" + messages[0]["content"]

    def encode(self, text, **kwargs):
        assert text == "CHAT:[SPEC]\nMy spec\n[END_SPEC]\n\n[PSEUDO_PROGRAM]My pseudo<eos><prefix_1><prefix_2>"
        assert kwargs == {"add_special_tokens": False}
        return [10, 11, compiler.PREFIX_TOKEN_START, compiler.PREFIX_TOKEN_START + 1]


def test_compiler_input_uses_exact_training_prompt_and_existing_prefix_tokens():
    assert compiler.build_compiler_input(PrefixTokenizer(), "My spec", "My pseudo", 2) == [
        10, 11, compiler.PREFIX_TOKEN_START, compiler.PREFIX_TOKEN_START + 1,
    ]


def test_compiler_rejects_tokenizer_without_trained_prefix_embeddings():
    tokenizer = PrefixTokenizer()
    tokenizer.convert_tokens_to_ids = lambda tokens: [0] * len(tokens)
    with pytest.raises(ValueError, match="trained prefix tokens"):
        compiler.build_compiler_input(tokenizer, "My spec", "My pseudo", 2)


def test_long_specs_leave_at_least_64_reference_tokens():
    assert compiler.pseudo_token_budget(100) == 1024
    assert compiler.pseudo_token_budget(5056) == 64
    with pytest.raises(ValueError, match="too long"):
        compiler.pseudo_token_budget(5057)
