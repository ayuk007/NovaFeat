"""Interactive CLI for the Agentic Feature Engineering Harness."""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from afe.agent.config_bundle import build_dependencies
from afe.graph.build_graph import build_graph
from afe.state.graph_state import GraphPhase, HarnessState

app = typer.Typer(add_completion=False, help="Agentic Feature Engineering Harness")
console = Console()

_HELP_TEXT = """
Available commands:
  /help      Show this help
  /state     Show current session phase and dataset info
  /plan      Show the current proposed plan
  /approve   Approve the current plan
  /reject    Reject the current plan
  /feedback  Provide feedback text
  /history   Show execution history
  /exit      Exit the session
"""


@app.command()
def run(
    csv_path: Path = typer.Argument(..., help="Path to the dataset CSV"), config_dir: str = "config"
) -> None:
    """Start an interactive feature-engineering session over CSV_PATH."""
    if not csv_path.exists():
        console.print(f"[red]File not found: {csv_path}[/red]")
        raise typer.Exit(code=1)

    df = pd.read_csv(csv_path)
    console.print(Panel(f"Loaded [bold]{csv_path.name}[/bold]: {df.shape[0]} rows x {df.shape[1]} cols"))

    deps = build_dependencies(df, config_dir=config_dir)
    graph = build_graph(deps)

    session_id = str(uuid.uuid4())
    thread_config = {"configurable": {"thread_id": session_id}}
    initial_state = HarnessState(session_id=session_id, phase=GraphPhase.INIT)

    result = graph.invoke(initial_state, config=thread_config)
    _interaction_loop(graph, thread_config, result)


@app.command()
def version() -> None:
    """Show the harness version."""
    console.print("afe-harness 0.1.0")


@app.command("list-transformations")
def list_transformations_cmd() -> None:
    """List all registered feature engineering transformations."""
    from afe.services.transformations import list_transformations

    for name in list_transformations():
        console.print(f"- {name}")


def _interaction_loop(graph, thread_config: dict, state: dict) -> None:
    while True:
        if "__interrupt__" in state:
            state = _handle_interrupt(graph, thread_config, state)
            continue

        phase = state.get("phase")
        if phase == GraphPhase.DONE or phase == GraphPhase.CANCELLED:
            console.print(Panel(f"Session ended (phase={phase})."))
            break

        command = Prompt.ask("[bold cyan]afe[/bold cyan]>")
        if command in ("/exit", "/quit"):
            console.print("Goodbye.")
            break
        if command == "/help":
            console.print(_HELP_TEXT)
        elif command == "/state":
            _print_state(state)
        elif command == "/history":
            for line in state.get("execution_history", []):
                console.print(f"- {line}")
        else:
            console.print("[yellow]Unknown command outside of an active prompt. Try /help.[/yellow]")


def _handle_interrupt(graph, thread_config: dict, state: dict):
    interrupt_obj = state["__interrupt__"][0]
    payload = interrupt_obj.value

    if payload.get("type") == "approval_request":
        _render_plan(payload)
        decision = Prompt.ask(
            "Approve this plan?", choices=["approve", "reject", "modify", "skip"], default="approve"
        )
        resume_value: dict = {"decision": decision}
        if decision == "modify":
            names = Prompt.ask("Comma-separated feature names to approve", default="")
            resume_value["approved_feature_names"] = [n.strip() for n in names.split(",") if n.strip()]
    elif payload.get("type") == "feedback_request":
        feedback = Prompt.ask(payload.get("prompt", "Feedback"), default="")
        resume_value = {"feedback": feedback}
    else:
        resume_value = {}

    from langgraph.types import Command

    return graph.invoke(Command(resume=resume_value), config=thread_config)


def _render_plan(payload: dict) -> None:
    table = Table(title=payload.get("summary", "Proposed Plan"))
    table.add_column("Feature")
    table.add_column("Operation")
    table.add_column("Source Columns")
    table.add_column("Rationale")
    table.add_column("Risk")
    for proposal in payload.get("proposals", []):
        table.add_row(
            proposal["name"],
            proposal["operation"],
            ", ".join(proposal["source_columns"]),
            proposal["rationale"],
            proposal["risk_level"],
        )
    console.print(table)


def _print_state(state: dict) -> None:
    console.print(f"Phase: {state.get('phase')}")
    metadata = state.get("dataset_metadata")
    if metadata:
        console.print(f"Dataset version: {metadata.version}, rows: {metadata.row_count}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
