"""LangGraph workflow.

use_lead=False: the fixed Phase 4 chain  profiler -> feature_engineer -> trainer -> evaluator.
use_lead=True (default, system 4): after the evaluator the lead picks an action; unless it says
finish, the action is applied and the chain runs again. The lead stops after 4 decisions.
`use_critic` and `disabled_agents` are filled in by Phases 6 and 8.
"""
from langgraph.graph import END, START, StateGraph

from src.agents.actions import apply_action
from src.agents.evaluator import run_evaluator
from src.agents.feature_engineer import run_feature_engineer
from src.agents.lead import run_lead
from src.agents.profiler import run_profiler
from src.agents.trainer import run_trainer
from src.graph.state import State

RUN_CONFIG = {"recursion_limit": 80}      # pass to graph.invoke: loops need more than the default 25 steps


def _route(state: State) -> str:
    return "end" if state.get("next_step") == "finish" else "apply_action"


def build_graph(use_critic: bool = False, disabled_agents: tuple = (),
                profiler_llm: bool = True, use_lead: bool = True):
    if use_critic or disabled_agents:
        raise NotImplementedError("use_critic and disabled_agents are added in Phases 6 to 8.")

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

    g = StateGraph(State)
    g.add_node("profiler", profiler_node)
    g.add_node("feature_engineer", feature_engineer_node)
    g.add_node("trainer", trainer_node)
    g.add_node("evaluator", evaluator_node)

    g.add_edge(START, "profiler")
    g.add_edge("profiler", "feature_engineer")
    g.add_edge("feature_engineer", "trainer")
    g.add_edge("trainer", "evaluator")

    if use_lead:
        g.add_node("lead", lead_node)
        g.add_node("apply_action", apply_action_node)
        g.add_edge("evaluator", "lead")
        g.add_conditional_edges("lead", _route, {"apply_action": "apply_action", "end": END})
        g.add_edge("apply_action", "feature_engineer")
    else:
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
    graph = build_graph(profiler_llm=args.llm, use_lead=not args.no_lead)
    out = graph.invoke(state, RUN_CONFIG)

    for name, r in out["results"].items():
        print(f"{name:20s} val ROC-AUC {r['val_roc_auc']}  rare recall {r.get('rare_class_recall')}")
    print("best model:", out["best_model"], f"({time.time() - t0:.1f}s)")
    for i, d in enumerate(out.get("decisions", []), 1):
        print(f"decision {i}: {d['action']} {d['columns']} [{d['source']}] {d['reason']}")