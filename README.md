# ProgramAsWeights Compiler

**Compile natural language specs into small neural functions on a single GPU.**

Describe a text-in/text-out function, compile it into a `.paw` file, and run it locally with the [Python SDK](https://github.com/programasweights/programasweights-python). Compilation uses the public Standard or Compact checkpoints. No hosted compilation service or API key is needed.

The complete compilation pipeline, including the learned LoRA mapper, is in [`compile.py`](compile.py).

## Quickstart

Compilation requires Python 3.10+ and an NVIDIA GPU with BF16 support (Ampere or newer). We recommend 16 GB of GPU memory. Models download from Hugging Face on the first run.

```bash
git clone https://github.com/programasweights/compiler.git
cd compiler
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Compile a function:

```bash
python compile.py \
  --spec "Classify if a message needs immediate attention or can wait. Return only 'immediate' or 'wait'." \
  --output email-triage.paw
```

Install the SDK to run the resulting file:

```bash
pip install programasweights --extra-index-url https://pypi.programasweights.com/simple/
```

```python
import programasweights as paw

fn = paw.function("./email-triage.paw")
print(fn("Urgent: the server is down!"))  # immediate
print(fn("Newsletter: spring picnic"))  # wait
```

The SDK downloads the shared interpreter once. Subsequent inference runs locally, including offline. You can copy the `.paw` file to another machine and use the same code.

## Choose a compiler

- **Standard** (default): higher accuracy, Qwen3 0.6B interpreter, approximately 22 MB per program and a shared 594 MB base model.
- **Compact**: GPT-2 124M interpreter, approximately 5 MB per program and a shared 134 MB base model. Uploaded Compact programs also run in the browser.

```bash
python compile.py \
  --compiler compact \
  --spec-file examples/email-triage.txt \
  --output email-triage-compact.paw
```

Use `--spec-file` for longer descriptions and examples. Select a GPU with `CUDA_VISIBLE_DEVICES`.

## Upload for remote inference

Export the original PEFT adapter alongside the `.paw` file:

```bash
python compile.py \
  --spec-file examples/email-triage.txt \
  --output email-triage.paw \
  --export-adapter email-triage-adapter
```

The export directory contains `adapter_model.safetensors`, `adapter_config.json`, and `manifest.json`. The manifest records the exact inference prompt and target runtime; upload all three together with [`upload.py`](upload.py).

Create an API key in [Settings](https://programasweights.com/settings), then upload:

```bash
export PAW_API_KEY=your_api_key
python upload.py email-triage-adapter
```

The command prints the program ID and Hub link. Programs are public by default; pass `--unlisted` to omit a program from the public Hub listing. If an upload is interrupted, rerun the same command to resume it.

Use the returned ID for remote inference:

```python
import programasweights as paw

with paw.function("YOUR_PROGRAM_ID", remote=True) as fn:
    print(fn("Urgent: the server is down!"))
```

Remote inference runs without downloading a local model. Add `--compiler compact` to the export command to upload a Compact program.

## How compilation works

1. A reference model turns the specification into a short pseudo-program with examples.
2. The trained compiler processes the specification, pseudo-program, and 64 learned prefix tokens.
3. The LoRA mapper converts the pooled compiler states into the interpreter's adapter weights.
4. The script exports the adapter and inference prompt as a `.paw` file, optionally retaining the original PEFT weights for upload.

The reference model and compiler run sequentially on one GPU. The interpreter is only needed when you run the compiled function.

On one H200, the example above compiled in about 10 seconds with Standard and 6 seconds with Compact, using about 8 GB of GPU memory. These timings start with the checkpoints already downloaded and include loading the models and exporting the program.

Both compiler checkpoints include the full compiler weights, tokenizer, learned prefix embeddings, and `lora_mapper.pt`:

- [Standard checkpoint](https://huggingface.co/programasweights/paw-4b-qwen3-0.6b) — release `20260407`.
- [Compact checkpoint](https://huggingface.co/programasweights/paw-4b-gpt2) — release `20260406`.
- [Reference model](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).

To use an already downloaded compiler checkpoint, pass `--checkpoint /path/to/checkpoint` and the matching `--compiler`.

## Citation

If you use this compiler in research, please cite Program-as-Weights, which introduces fuzzy-function programming and the fast compiler.

**[Program-as-Weights: A Programming Paradigm for Fuzzy Functions](https://arxiv.org/abs/2607.02512)**

NeurIPS 2026 · [Preprint](https://arxiv.org/abs/2607.02512)

```bibtex
@inproceedings{zhang2026programasweights,
  title = {{Program-as-Weights: A Programming Paradigm for Fuzzy Functions}},
  author = {Zhang, Wentao and Hotsko, Liliana and Kim, Woojeong and Nie, Pengyu and Shieber, Stuart and Deng, Yuntian},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year = {2026},
  url = {https://arxiv.org/abs/2607.02512}
}
```

### Related work

Compile by Training builds on PAW with teacher-generated examples and task-specific training, using the same program format and runtime.

**[Compile by Training: Turning Natural-Language Specifications into Local Neural Functions](https://arxiv.org/abs/2609.04199)**

```bibtex
@inproceedings{paw_2609_04199,
  title = {{Compile by Training: Turning Natural-Language Specifications into Local Neural Functions}},
  author = {Deng, Yuntian and Nie, Pengyu and Shieber, Stuart},
  booktitle = {EMNLP 2026 System Demonstrations},
  year = {2026},
  url = {https://arxiv.org/abs/2609.04199}
}
```

## Learn more

- [ProgramAsWeights paper](https://arxiv.org/abs/2607.02512)
- [Website and examples](https://programasweights.com/docs#examples)
- [Python SDK](https://github.com/programasweights/programasweights-python)
- [Browser SDK](https://github.com/programasweights/programasweights-js)

Model licenses are provided by their respective publishers.
