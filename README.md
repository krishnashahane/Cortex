# Cortex

Autonomous, reproducible ML research loop with a local dashboard.

Cortex executes real scikit-learn experiments through a LangGraph workflow:

research -> hypothesize -> plan -> train -> evaluate -> critique -> repeat -> report

It works offline by default. Anthropic and Google Gemini are optional reasoning providers.

## Install

    git clone https://github.com/krishnashahane/Cortex.git
    cd Cortex
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

## Run

    python run.py research
    python run.py research --goal "Find a strong tabular classifier" --iters 4
    python run.py serve

Dashboard: http://127.0.0.1:8000

The dashboard is loopback-only because its API starts compute-heavy research jobs and has no remote authentication.

## Optional LLMs

    export CORTEX_LLM_PROVIDER=auto
    export ANTHROPIC_API_KEY="..."

or:

    export CORTEX_LLM_PROVIDER=gemini
    export GEMINI_API_KEY="..."

Provider values: auto, anthropic, gemini, offline.

Failed provider calls fall back to the deterministic offline path.

Defaults:
- Anthropic model: claude-opus-4-8
- Gemini model: gemini-3.8-flash

Override with CORTEX_ANTHROPIC_MODEL and CORTEX_GEMINI_MODEL.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| CORTEX_MAX_ITERATIONS | 8 | Research budget, capped at 30 |
| CORTEX_IMPROVEMENT_THRESHOLD | 0.001 | Relative improvement threshold |
| CORTEX_PATIENCE | 2 | Low-improvement rounds before stopping |
| CORTEX_DATASET | breast_cancer | breast_cancer, wine, digits, iris |
| CORTEX_PRIMARY_METRIC | accuracy | accuracy, f1, precision, recall, roc_auc |
| CORTEX_RANDOM_STATE | 42 | Reproducible seed |
| CORTEX_LLM_PROVIDER | auto | auto, anthropic, gemini, offline |
| CORTEX_DATA_DIR | ./data | SQLite and Chroma data |
| CORTEX_REPORTS_DIR | ./reports | Markdown report directory |

## Architecture

    CEO
      |
      +-- continue --> Research -> Hypothesis -> Plan -> Train
      |                                      |          |
      |                                      +---- Evaluate
      |                                               |
      |                                            Critic
      |                                               |
      +-- terminate ------------------------------> Report

The CEO owns deterministic termination. The Trainer executes a real ML pipeline; results are never simulated.

## Security and data hygiene

Runtime data is intentionally ignored by Git:

    data/cortex.db
    data/chroma/
    reports/run_*.md

API keys are read only from environment variables. They are not persisted to SQLite, Chroma metadata, or reports.

The API validates request size and iteration limits, the dashboard only binds to loopback, and concurrent research jobs are bounded.

## Development

    python -m pip install -r requirements.txt -r requirements-dev.txt
    python -m compileall -q cortex run.py tests
    ruff check .
    pytest -q
    pip-audit

CI runs the same checks plus a one-iteration offline smoke research run.

## License

MIT
