"""Systems 4 and 5: runs the LangGraph team and returns one result row.

System 4 = agents_no_critic (lead on, critic off).
System 5 = agents_with_critic (identical, critic on).
"""
from src.agents.feature_engineer import select_xy
from src.evaluation.metrics import score_on_test
from src.graph.workflow import RUN_CONFIG, build_graph
from src.llm.client import get_stats, reset_stats
from src.utils.cost_tracker import CostTracker
from src.utils.data import ROOT, TARGET_COL
from src.utils.seeds import set_seed
from src.agents.explainer import run_explainer
from src.agents.reporter import run_reporter

PROFILER_USES_LLM = False 
EXPLAINER_USES_LLM = False   # template explanation until the LLM quota is sorted out  # set True if you want the LLM profiler summary in benchmark runs


def _initial_state(name: str, seed: int) -> dict:
    base = ROOT / "data" / "processed" / f"{name}_seed{seed}"
    return {
        "dataset_name": name,
        "goal": "predict the target column",
        "target": TARGET_COL,
        "seed": seed,
        "train_path": f"{base}_train.csv",
        "val_path": f"{base}_val.csv",
    }


def _run(system: str, name: str, seed: int, use_critic: bool) -> dict:
    tracker = CostTracker()
    reset_stats()
    row = {"system": system, "dataset": name, "seed": seed}
    try:
        set_seed(seed)
        graph = build_graph(profiler_llm=PROFILER_USES_LLM, use_lead=True, use_critic=use_critic)
        out = graph.invoke(_initial_state(name, seed), RUN_CONFIG)

        best = out["best_model"]
        model = out["models"][best]
        val_auc = out["results"][best]["val_roc_auc"]

        def predict_fn(X_test):
            df = X_test.copy()
            df[out["target"]] = 0
            X, _ = select_xy(df, out["features"], out["target"], out.get("extra_features"))
            return model.predict_proba(X)[:, 1]

        test = score_on_test(name, seed, predict_fn)

        out["test_score"] = round(test["roc_auc"], 4)
        out = run_explainer(out, use_llm=EXPLAINER_USES_LLM)
        out = run_reporter(out)
        report_dir = ROOT / "results" / "raw" / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / f"{system}_{name}_seed{seed}.md").write_text(out["report"], encoding="utf-8")

        decisions = out.get("decisions", [])
        row.update(
            best_model=best,
            val_score=round(val_auc, 4),
            test_score=round(test["roc_auc"], 4),
            score_drop=round(val_auc - test["roc_auc"], 4),
            test_f1=round(test["f1"], 4),
            test_balanced_accuracy=round(test["balanced_accuracy"], 4),
            test_rare_class_recall=round(test["rare_class_recall"], 4),
            lead_actions=[d["action"] for d in decisions if d.get("source") != "critic"],
            lead_sources=[d.get("source") for d in decisions if d.get("source") != "critic"],
            lead_fallbacks=sum(d.get("source") == "fallback" for d in decisions),
            dropped=out.get("dropped", {}),
        )
        if use_critic:
            row.update(
                flags_raised=[f["check"] for f in out.get("critic_flags", [])],   # flags of the final pass
                critic_drops=[d["columns"] for d in decisions if d.get("source") == "critic"],
                critic_sendbacks=out.get("critic_loops", 0),
                critic_source=(out.get("critic_verdict") or {}).get("source"),
            )
    except Exception as e:
        row.update(crashed=True, error=f"{type(e).__name__}: {e}")

    s = get_stats()
    tracker.llm_calls, tracker.tokens = s["calls"], s["tokens"]
    row.update(tracker.summary())
    return row


def run_agents_no_critic(name: str, seed: int) -> dict:
    return _run("agents_no_critic", name, seed, use_critic=False)


def run_agents_with_critic(name: str, seed: int) -> dict:
    return _run("agents_with_critic", name, seed, use_critic=True)


if __name__ == "__main__":
    print(run_agents_with_critic("credit-g", 0))