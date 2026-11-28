"""LangGraph workflow.

use_lead=False: the fixed Phase 4 chain  profiler -> feature_engineer -> trainer -> evaluator.
use_lead=True (default, system 4): after the evaluator the lead picks an action; unless it says
finish, the action is applied and the chain runs again. The lead stops after 4 decisions.
use_critic=True (system 5): the critic runs after every evaluator pass. On send_back with
droppable columns, critic_fix drops them and the chain reruns (max 3 send-backs).
`disabled_agents` is filled in by Phase 8.
"""
from langgraph.graph import END, START, StateGraph

from src.agents.actions import apply_action
from src.agents.critic import run_critic
from src.agents.evaluator import run_evaluator
from src.agents.feature_engineer import run_feature_engineer
from src.agents.lead import run_lead
from src.agents.profiler import run_profiler
from src.agents.trainer import run_trainer
from src.graph.state import State

RUN_CONFIG = {"recursion_limit": 80}      # pass to graph.invoke: loops need more than the default 25 steps

FIXABLE_CHECKS = ("single_feature_leakage", "id_like_columns")


def _route(state: State) -> str:
    return "end" if state.get("next_step") == "finish" else "apply_action"


def critic_fix_columns(state: State) -> list:
    """Columns named by serious leakage/ID flags that can be dropped. Never drops every feature."""
    features = list(state.get("features", []))
    cols = []
    for f in state.get("critic_flags", []):
        if f["severity"] == "serious" and f["check"] in FIXABLE_CHECKS:
            cols += [c for c in f["details"] if c in features and c not in cols]
    return [] if len(cols) >= len(features) else cols


def build_graph(use_critic: bool = False, disabled_agents: tuple = (),
                profiler_llm: bool = True, use_lead: bool = True, critic_llm: bool = True):
    if disabled_agents:
        raise NotImplementedError("disabled_agents is added in Phase 8.")

    def profiler_node(state: State) -> State:
        return run_profiler(state, use_llm=profiler_llm)

    def feature_engineer_node(state: State) -> State:
        return run_feature_engineer(state)

    def trainer_node(state: State) -> State:
        return run_trainer(state)

    def evaluator_node(state: State) -> State:
        return run_evaluator(state)

    def lead_node(state: State) -> State:
        return run_lead(state)

    def apply_action_node(state: State) -> State:
        return apply_action(state)

    def critic_node(state: State) -> State:
        return run_critic(state, use_llm=critic_llm)

    def critic_fix_node(state: State) -> State:
        cols = critic_fix_columns(state)
        why = "; ".join(f"{f['check']} {f['details']}" for f in state["critic_flags"]
                        if f["severity"] == "serious" and f["check"] in FIXABLE_CHECKS)
        decision = {"action": "drop_columns", "columns": cols,
                    "reason": ("critic: " + why)[:300], "source": "critic"}
        state = {**state, "next_step": "drop_columns",
                 "decisions": state.get("decisions", []) + [decision]}
        return apply_action(state)

    def route_critic(state: State) -> str:
        if state.get("next_step") == "send_back" and critic_fix_columns(state):
            return "critic_fix"
        return "lead" if use_lead else "end"

    g = StateGraph(State)
    g.add_node("profiler", profiler_node)
    g.add_node("feature_engineer", feature_engineer_node)
    g.add_node("trainer", trainer_node)
    g.add_node("evaluator", evaluator_node)

    g.add_edge(START, "profiler")
    g.add_edge("profiler", "feature_engineer")
    g.add_edge("feature_engineer", "trainer")
    g.add_edge("trainer", "evaluator")

    if use_critic:
        g.add_node("critic", critic_node)
        g.add_node("critic_fix", critic_fix_node)
        g.add_edge("evaluator", "critic")
        g.add_edge("critic_fix", "feature_engineer")
        targets = {"critic_fix": "critic_fix", "end": END}
        if use_lead:
            targets["lead"] = "lead"
        g.add_conditional_edges("critic", route_critic, targets)

    if use_lead:
        g.add_node("lead", lead_node)
        g.add_node("apply_action", apply_action_node)
        if not use_critic:
            g.add_edge("evaluator", "lead")
        g.add_conditional_edges("lead", _route, {"apply_action": "apply_action", "end": END})
        g.add_edge("apply_action", "feature_engineer")
    elif not use_critic:
        g.add_edge("evaluator", END)
    return g.compile()


# ---------- checkpoint runner ----------

if __name__ == "__main__":
    import argparse
    import time

    from src.evaluation.metrics import TARGET_COL
    from src.utils.helpers import PROJECT_ROOT

    ap = argparse.ArgumentParser(description="Run the agent graph on one dataset and seed.")
    ap.add_argument("name", help="dataset name, for example credit-g")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--llm", action="store_true", help="let the profiler call the LLM for its summary")
    ap.add_argument("--no-lead", action="store_true", help="fixed Phase 4 chain, no lead")
    ap.add_argument("--critic", action="store_true", help="system 5: add the critic")
    ap.add_argument("--no-critic-llm", action="store_true", help="critic uses code verdicts only")
    args = ap.parse_args()

    base = PROJECT_ROOT / "data" / "processed" / f"{args.name}_seed{args.seed}"
    state: State = {
        "dataset_name": args.name,
        "goal": "predict the target column",
        "target": TARGET_COL,
        "seed": args.seed,
        "train_path": f"{base}_train.csv",
        "val_path": f"{base}_val.csv",
    }
    t0 = time.time()
    graph = build_graph(profiler_llm=args.llm, use_lead=not args.no_lead,
                        use_critic=args.critic, critic_llm=not args.no_critic_llm)
    out = graph.invoke(state, RUN_CONFIG)

    for name, r in out["results"].items():
        print(f"{name:20s} val ROC-AUC {r['val_roc_auc']}  rare recall {r.get('rare_class_recall')}")
    print("best model:", out["best_model"], f"({time.time() - t0:.1f}s)")
    for i, d in enumerate(out.get("decisions", []), 1):
        print(f"decision {i}: {d['action']} {d['columns']} [{d['source']}] {d['reason']}")
    print("critic verdict:", out.get("critic_verdict"), "| flags:", [f["check"] for f in out.get("critic_flags", [])])
    print("dropped:", out.get("dropped"))