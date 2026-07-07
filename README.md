# Agentic Feature Engineering Harness — Phase 1

An interactive, LangGraph-based CLI that performs feature engineering on tabular
datasets through an agentic, human-in-the-loop workflow — **without ever handing
the LLM the raw dataset**. The agent only ever sees schema, statistical profiles,
correlations, and small human-approved previews.

This repository is being built **incrementally, in production-quality phases**.
This document describes what **Phase 1** actually delivers, what is intentionally
deferred, and how the two connect architecturally so later phases are additive,
not rewrites.

---

## What Phase 1 delivers (real, tested, working)

- **Configuration system** (`src/afe/config`): every tunable lives in
  `config/*.yaml`, composed into a single validated `RootConfig` (Pydantic),
  with environment-variable overrides (`AFE__SECTION__FIELD=value`) and
  `.env` support. No hardcoded settings anywhere in application code.
- **Typed domain models** (`src/afe/models`): dataset schema/profile/correlation
  models, PII assessment models, tool metadata, and typed inter-agent messages
  (`PlanMessage`, `ExecutionMessage`, `FeedbackMessage`, `ApprovalMessage`, ...).
- **Strongly typed LangGraph state** (`src/afe/state/graph_state.py`): a single
  Pydantic `HarnessState` checkpointed by LangGraph across the whole session.
  It never contains a DataFrame — only metadata and summaries.
- **Data Access Service** (`src/afe/services/data_access_service.py`): the
  *only* gateway to the dataset. Implements schema/profile/correlation/preview/
  transform/rollback against an in-memory pandas engine, behind the
  `DataAccessService` interface so a Polars/DuckDB/Dask engine can be dropped in
  later without touching any agent or graph code.
- **PII detection pipeline (stage 1 of N)**: a regex/name-heuristic detector
  (`RegexPIIDetector`) composed through a `PIIAggregator` that already
  implements the "multiple detectors vote, confidence aggregated" architecture
  the full spec calls for. NER and LLM-classification stages are additional
  `PIIDetector` implementations to be appended to the same aggregator list —
  no change to the aggregator or the Data Access Service is required.
- **A real LangGraph graph** (`src/afe/graph/build_graph.py`), not a script:
  `dataset_analyzer → planner → human_approval →(conditional)→ feature_executor
  → summary_generator → feedback_processor →(conditional, loops)→ planner`,
  with `interrupt()`-based human approval and a bounded replanning loop.
- **Model factory** (`src/afe/services/model_factory.py`): the single call
  site for `langchain.chat_models.init_chat_model`. Ten independently
  configurable model roles (planner, code_generation, pii_detection,
  summarization, critique, validation, context_compression, feedback,
  tool_selection, reasoning) are already defined in `config/model.yaml` —
  wiring a role to a different provider/model is a YAML edit, not a code change.
- **Context builder** (`src/afe/services/context_builder.py`): priority-ordered,
  token-budgeted context assembly from state, dropping low-priority sections
  first when the budget is exceeded.
- **Registered feature transformations** (`src/afe/services/transformations.py`):
  fill_missing, log_transform, ratio, standard_scale, one_hot_encode, binning,
  date_features — a Strategy-pattern registry that new transformations plug
  into via `@register("name")`.
- **Interactive CLI** (`src/afe/cli/main.py`): Typer + Rich, renders the plan
  as a table, drives the approve/reject/modify/skip loop and feedback prompt
  against the graph's interrupts.
- **37 passing tests** across config, profiling, PII, data access, transformations,
  state/messages, and full graph integration (mocked model) — including the
  approval, rejection, and feedback→replan loops. `mypy` and `ruff` both clean.

## What Phase 1 intentionally does not yet include

These are named explicitly in the spec and are real, separate subsystems —
building them as stubs would violate the "no fake implementations" requirement,
so they are deferred to their own phases rather than faked here:

- Dynamic tool/code generation pipeline (Code Generator Agent → Static/AST
  Validator → Security Validator → Policy Validator → Sandbox Executor).
  `Tool` interface and `ToolCapability`/`ToolResult` models already exist
  (`src/afe/interfaces/tool.py`, `src/afe/models/tool_metadata.py`) so this
  phase plugs in without touching the graph.
- Sandboxed execution backends (subprocess/Docker/Podman/gVisor).
- NER- and LLM-classification PII stages, and a local HuggingFace PII model.
- Local-model fallback flow (Ollama/llama.cpp/Transformers/vLLM/LM Studio).
- Polars/DuckDB/Dask/Spark engines and the large-dataset sampling-strategy
  prompt (`DataAccessService` is already engine-agnostic; only new
  implementations are needed).
- Versioned dataset checkpointing beyond in-memory version history
  (persistent checkpoints, branching).
- Runtime `/config` CLI commands, rich token-usage export (JSON/CSV reports).
- Structured JSON logging, retry/circuit-breaker policies, plugin persistence.

## Architecture at a glance

```
CLI (Typer/Rich)
   │
   ▼
AgentDependencies (DI container)  ──uses──▶  RootConfig (YAML + env)
   │
   ▼
LangGraph graph (HarnessState, checkpointed)
   │
   ├─ dataset_analyzer ──▶ DataAccessService ──▶ pandas engine (private)
   ├─ planner           ──▶ ModelFactory (init_chat_model) + ContextBuilder
   ├─ human_approval     ──▶ interrupt() ──▶ CLI prompt
   ├─ feature_executor   ──▶ DataAccessService.apply_transformation (Strategy registry)
   ├─ summary_generator
   └─ feedback_processor ──▶ interrupt() ──▶ CLI prompt ──▶ loops back to planner
```

See `docs/architecture.md` for the Mermaid diagrams (component, graph, and
sequence).

## Quickstart

```bash
pip install -e ".[dev]"
export OPENAI_API_KEY=sk-...   # or configure a different provider in config/model.yaml
afe run path/to/dataset.csv
```

Without an API key, everything except the LLM-backed planner call runs (data
access, PII detection, profiling, transformation execution, config loading) —
see `tests/test_graph.py` for how the graph is exercised end-to-end with a
mocked model.

## Development

```bash
pip install -e ".[dev]"
pytest                 # 37 tests
ruff check src/
mypy src/afe --ignore-missing-imports
black src/ tests/ --line-length 110
```

## Configuration

Every section under `config/*.yaml` maps to a Pydantic model in
`src/afe/config/schemas.py`. Override any field via environment variable:

```bash
export AFE__MODEL__PLANNER__PROVIDER=anthropic
export AFE__MODEL__PLANNER__MODEL=claude-sonnet-4-6
export AFE__SECURITY__BLOCKED_COLUMNS='["ssn", "credit_card_number"]'
```

## Extensibility points (already wired for later phases)

| Extension point | Interface | How to extend |
|---|---|---|
| New dataset engine | `afe.interfaces.data_access.DataAccessService` | Implement the ABC; swap in `AgentDependencies` |
| New model provider | `model.yaml` role config | Change `provider`/`model`; no code change |
| New PII stage | `afe.interfaces.pii_detector.PIIDetector` | Add to the `PIIAggregator` detector list |
| New transformation | `afe.services.transformations.register` | `@register("name")` decorated function |
| New tool | `afe.interfaces.tool.Tool` | Implement the ABC; register with the (future) Tool Registry |
| New graph node | LangGraph `add_node` | Add node + wire conditional edges in `build_graph.py` |
