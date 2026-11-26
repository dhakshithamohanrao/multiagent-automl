"""LangGraph workflow.

Phase 4: a fixed chain  profiler -> feature_engineer -> trainer -> evaluator.
`use_critic` and `disabled_agents` are accepted now so later phases can fill them in
(lead in Phase 5, critic in Phase 6, ablations in Phase 8).
"""
from langgraph.graph import END, START, StateGraph

from src.agents.evaluator import run_evaluator
from src.agents.feature_engineer import run_feature_engineer
from src.agents.profiler import run_profiler
from src.agents.trainer import run_trainer
from src.graph.state import State


def build_graph(use_critic: bool = False, disabled_agents: tuple = (), profiler_llm: bool = True):
    if use_critic or disabled_agents:
        raise NotImplementedError("use_critic and disabled_agents are added in Phases 5 to 8.")

    def profiler_node(state: State) -> State:
        return run_profiler(state, use_llm=profiler_llm)

    def feature_engineer_node(state: State) -> State:
        return run_feature_engineer(state)

    def trainer_node(state: State) -> State:
        return run_trainer(state)

    def evaluator_node(state: State) -> State:
        return run_evaluator(state)

    g = StateGraph(State)
    g.add_node("profiler", profiler_node)
    g.add_node("feature_engineer", feature_engineer_node)
    g.add_node("trainer", trainer_node)
    g.add_node("evaluator", evaluator_node)

    g.add_edge(START, "profiler")
    g.add_edge("profiler", "feature_engineer")
    g.add_edge("feature_engineer", "trainer")
    g.add_edge("trainer", "evaluator")
    g.add_edge("evaluator", END)
    return g.compile()


# ---------- checkpoint runner ----------

if __name__ == "__main__":
    import argparse
    import time

    from src.evaluation.metrics import TARGET_COL
    from src.utils.helpers import PROJECT_ROOT

    ap = argparse.ArgumentParser(description="Step 4.7 checkpoint: run the whole chain through LangGraph.")
    ap.add_argument("name", help="dataset name, for example credit-g")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--llm", action="store_true", help="let the profiler call the LLM for its summary")
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
    out = build_graph(profiler_llm=args.llm).invoke(state)
    for name, r in out["results"].items():
        print(f"{name:20s} val ROC-AUC {r['val_roc_auc']}")
    print("best model:", out["best_model"], f"({time.time() - t0:.1f}s)")