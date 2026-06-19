# Cortex

> Autonomous Multi-Agent ML Research Scientist

Cortex is an autonomous AI research system that mimics a team of machine learning researchers. It continuously reads papers, generates hypotheses, runs experiments, evaluates results, critiques itself, and iterates until convergence.

## Features

* Multi-agent architecture
* Autonomous research loops
* Paper understanding and knowledge retrieval
* Hypothesis generation
* Experiment planning and execution
* Training and evaluation pipelines
* Self-critique and optimization
* Long-term memory with vector databases
* Experiment tracking and report generation
* Modular and production-ready design

## Agents

### CEO Agent

Defines objectives and milestones.

### Paper Reader Agent

Reads papers, repositories and documentation.

### Hypothesis Agent

Generates new model ideas and improvements.

### Experiment Agent

Creates training configurations and schedules.

### Trainer Agent

Runs experiments and model training.

### Evaluator Agent

Measures metrics and compares results.

### Critic Agent

Analyzes weaknesses and proposes improvements.

### Report Writer Agent

Produces reports and documentation.

## Workflow

```text
Research
↓
Read Papers
↓
Generate Hypotheses
↓
Design Experiments
↓
Train Models
↓
Evaluate Results
↓
Critique
↓
Improve
↓
Repeat
```

## Architecture

```text
CEO
 ↓
Paper Reader
 ↓
Hypothesis Generator
 ↓
Experiment Planner
 ↓
Trainer
 ↓
Evaluator
 ↓
Critic
 ↓
Report Writer
 ↓
Memory
 ↓
Loop
```

## Tech Stack

* Python
* LangGraph
* FastAPI
* ChromaDB
* PostgreSQL
* Redis
* Docker
* PyTorch
* HuggingFace Transformers
* Ollama / Claude API / OpenAI API

## Example Goals

* Improve object detection models
* Optimize LLM architectures
* Discover better training strategies
* Evaluate model variants automatically
* Generate reproducible reports

## Directory Structure

```text
cortex/

agents/
memory/
papers/
experiments/
models/
reports/
tests/
src/
configs/
docs/
```

## Vision

Cortex aims to become an autonomous ML scientist capable of conducting iterative research and accelerating scientific discovery through agentic engineering.

## Future

* Distributed agents
* Reinforcement learning optimization
* Multi-modal research
* Automatic paper publishing
* Self-improving architectures
* Large-scale experiment orchestration

---

Built with agentic engineering, not vibe coding.
