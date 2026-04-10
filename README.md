# Probe — AI red team engine

Automated adversarial testing for LLMs and agentic AI systems.

## Setup

```bash
# 1. create virtualenv
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. configure API keys
cp .env.example .env
# edit .env and add your OPENAI_API_KEY
```

## Run

```bash
# attack the built-in AcmeShop demo target
python run.py

# attack your own system prompt
python run.py --system-prompt ./my_prompt.txt --target "my-agent"

# faster run: 1 round, 5 attacks per category, no agentic sim
python run.py --rounds 1 --attacks 5 --no-agentic

# full run: 3 rounds, 10 attacks per category + agentic simulation
python run.py --rounds 3 --attacks 10
```

## Output

- Terminal: live findings as they are discovered
- `output/run_{id}.json`: full structured report

## Architecture

```
probe/
├── engine/
│   ├── attacker.py    — generates adversarial prompts (attacker LLM)
│   ├── target.py      — fires prompts at the model under test
│   ├── judge.py       — scores responses (LLM judge + deterministic verifier)
│   ├── loop.py        — adaptive multi-round attack loop
│   └── agentic.py     — multi-turn agentic session simulator
├── models/
│   └── finding.py     — Pydantic data models (Finding, Run, Severity)
├── suites/
│   └── basic.yaml     — seed attack library (50 prompts, 6 categories)
└── run.py             — entrypoint
```

## Using a local model (air-gapped)

Install [Ollama](https://ollama.ai), pull a model, then set in `.env`:

```
TARGET_MODEL=ollama/llama3:8b
```

No data leaves your machine when targeting a local model.
# probe
