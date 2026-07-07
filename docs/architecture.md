# Architecture

## Component diagram

```mermaid
graph TD
    CLI[Interactive CLI - Typer/Rich]
    DI[AgentDependencies - DI container]
    CFG[RootConfig - YAML + env]
    GRAPH[LangGraph StateGraph]
    STATE[HarnessState - Pydantic, checkpointed]
    DAS[DataAccessService]
    ENGINE[Pandas engine - private to DAS]
    PII[PIIAggregator]
    MF[ModelFactory - init_chat_model]
    CB[ContextBuilder]
    XFORM[Transformation Registry]

    CLI --> DI
    DI --> CFG
    DI --> DAS
    DI --> MF
    DI --> CB
    DI --> GRAPH
    GRAPH --> STATE
    GRAPH --> DAS
    GRAPH --> MF
    GRAPH --> CB
    DAS --> ENGINE
    DAS --> PII
    DAS --> XFORM
```

## Graph structure

```mermaid
graph TD
    START([start]) --> analyzer[dataset_analyzer]
    analyzer --> planner[planner]
    planner --> approval[human_approval - interrupt]
    approval -->|approve/modify| executor[feature_executor]
    approval -->|skip| summary[summary_generator]
    approval -->|reject/cancel| exit[exit_node]
    executor --> summary
    summary --> feedback[feedback_processor - interrupt]
    feedback -->|constraint given, replan_count<=3| planner
    feedback -->|no feedback| exit
    exit --> END([end])
```

## Approval sequence

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Graph as LangGraph
    participant Planner
    participant DAS as DataAccessService

    Graph->>DAS: get_metadata/get_schema/get_all_column_profiles/get_correlation_summary
    DAS-->>Graph: statistical summaries only
    Graph->>Planner: build_context(state) + invoke(planner model)
    Planner-->>Graph: PlanMessage (proposed features)
    Graph->>CLI: interrupt(approval_request)
    CLI->>User: render plan table, prompt decision
    User-->>CLI: approve / reject / modify / skip
    CLI->>Graph: Command(resume={"decision": ...})
    alt approved or modified
        Graph->>DAS: apply_transformation(...) per approved proposal
        DAS-->>Graph: updated DatasetMetadata (new version)
    end
    Graph->>CLI: interrupt(feedback_request)
    CLI->>User: prompt for feedback
    User-->>CLI: feedback text or blank
    CLI->>Graph: Command(resume={"feedback": ...})
    alt feedback given
        Graph->>Planner: replan with new constraint
    else blank
        Graph->>Graph: exit_node -> DONE
    end
```

## Data-privacy enforcement points

1. `PandasDataAccessService._build_schema` classifies every column via
   `PIIAggregator` at construction and after every transformation; PII/blocked
   columns are flagged in `ColumnSchema.is_pii` / `is_blocked`.
2. `DataAccessService._assert_visible` (called by `get_column_profile`,
   `get_correlation_summary`, `request_preview`) raises `PolicyViolationError`
   for any blocked/PII column — this is enforced in code, not by prompting the
   LLM to "be careful."
3. `ContextBuilder.build` only ever reads from `HarnessState`, which itself
   only ever receives `DatasetMetadata`/`ColumnProfile`/`CorrelationSummary`/
   `PreviewResult` objects — there is no code path by which a raw DataFrame
   can enter the state that feeds the LLM prompt.
