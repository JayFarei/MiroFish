# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MiroFish is an AI prediction engine using multi-agent swarm intelligence. It builds a "parallel digital world" from seed material (news, policy drafts, financial data, novels), simulates thousands of AI agents with independent personalities interacting on social platforms (Twitter/Reddit via CAMEL-AI OASIS), and generates prediction reports. Incubated by Shanda Group.

## Commands

```bash
# Full setup (Node + Python)
npm run setup:all

# Development (both frontend:3000 + backend:5001)
npm run dev

# Backend only
cd backend && uv run python run.py

# Frontend only
cd frontend && npm run dev

# Build frontend for production
npm run build

# Run tests
cd backend && uv run pytest
cd backend && uv run pytest tests/test_specific.py -k "test_name"

# Install Python deps
cd backend && uv sync

# Docker
cp .env.example .env  # fill in keys first
docker compose up -d
```

## Architecture

### 5-Stage Pipeline

The entire system is a sequential pipeline matching the UI stepper (Steps 1-5):

1. **Graph Building** — Upload documents → LLM extracts ontology (entity/relationship types) → text chunked and ingested into Zep Cloud knowledge graph (GraphRAG)
2. **Environment Setup** — Read entities from Zep graph → LLM generates OASIS agent personas (CSV for Twitter, JSON for Reddit) + simulation parameters
3. **Simulation** — Spawns subprocess(es) via `backend/scripts/` (Twitter, Reddit, or parallel). IPC for pause/resume/stop. Agent actions logged, Zep memory updated in real-time
4. **Report Generation** — `ReportAgent` uses ReACT loop with Zep retrieval tools (`InsightForge`, `PanoramaSearch`, `QuickSearch`) to gather evidence and produce structured Markdown
5. **Deep Interaction** — Chat with any simulated agent or the ReportAgent for follow-up analysis

### Tech Stack

- **Backend**: Python 3.11+ / Flask 3.x / OpenAI SDK (any OpenAI-compatible LLM, default: Alibaba Qwen-plus) / Zep Cloud (GraphRAG + memory) / CAMEL-AI OASIS (simulation)
- **Frontend**: Vue 3.5 (Composition API) / Vite 7 / D3.js (graph visualization) / Axios
- **Package management**: `uv` for Python, `npm` for Node

### Key Abstractions

- **`ProjectManager`** (`backend/app/models/project.py`) — File-based project persistence at `backend/uploads/projects/<id>/project.json`. No traditional database.
- **`TaskManager`** (`backend/app/models/task.py`) — In-memory thread-safe singleton tracking async task progress (graph build, report generation)
- **`SimulationManager`** (`backend/app/services/simulation_manager.py`) — Orchestrates simulation preparation (reads Zep, generates profiles + config)
- **`SimulationRunner`** (`backend/app/services/simulation_runner.py`) — Manages simulation subprocess lifecycle and IPC
- **`ReportAgent`** (`backend/app/services/report_agent.py`) — ReACT-style LLM agent with Zep retrieval toolset
- **`LLMClient`** (`backend/app/utils/llm_client.py`) — OpenAI SDK wrapper, supports primary + optional "boost" LLM config

### API Structure

Three Flask blueprints registered under `/api`:
- `/api/graph/*` — Project CRUD, ontology generation, graph building, task status
- `/api/simulation/*` — Create, prepare, start/stop, status polling
- `/api/report/*` — Generate, status, download, chat

Frontend API modules in `frontend/src/api/` mirror these blueprints. Vite proxies `/api` to `localhost:5001`.

### Concurrency Model

Long-running operations (graph building, simulation, report generation) run in background Python threads. Progress is tracked via `TaskManager` and polled by the frontend.

## Environment Variables

Required in `.env` at project root (see `.env.example`):
- `LLM_BACKEND` — `openai` (default) or `claude_code` (Claude Code headless mode)
- `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL_NAME` — Primary LLM (used when `LLM_BACKEND=openai`)
- `CLAUDE_CODE_BIN`, `CLAUDE_CODE_MODEL`, `CLAUDE_CODE_TIMEOUT` — Claude Code settings (used when `LLM_BACKEND=claude_code`)
- `ZEP_API_KEY` — Zep Cloud API key
- Optional: `LLM_BOOST_API_KEY`, `LLM_BOOST_BASE_URL`, `LLM_BOOST_MODEL_NAME` — Secondary faster/cheaper LLM

## Language Note

Most Python docstrings, comments, log messages, system prompts, API error messages, and Vue UI labels are written in Chinese. The primary README is Chinese (`README.md`), with an English version at `README-EN.md`.
