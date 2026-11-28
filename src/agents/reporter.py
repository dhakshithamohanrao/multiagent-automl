"""Reporter agent: writes the Markdown report from the state. Pure code, no LLM.

Reads dataset_name, goal, target, profile, dropped, decisions, results, best_model,
critic_flags, critic_verdict, explanation, and optionally test_score (the final test result).
Writes only `report`. Every number is copied from the state, never computed by an LLM.
"""


def _fmt(v) -> str:
    return "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def build_report(state: dict) -> str:
    p = state.get("profile", {})
    L = [f"# Report: {state.get('dataset_name')}", "",
         f"**Goal:** {state.get('goal')}  ", f"**Target column:** {state.get('target')}  ",
         f"**Best model:** {state.get('best_model')}", "", "## Data profile", ""]
    L += [f"- Rows: {_fmt(p.get('n_rows'))}",
          f"- Duplicate rows: {_fmt(p.get('duplicate_rows'))}",
          f"- Class balance: {_fmt(p.get('class_balance'))}", ""]

    L += ["## Columns dropped", ""]
    dropped = state.get("dropped", {})
    L += [f"- `{c}`: {r}" for c, r in dropped.items()] or ["- none"]
    L.append("")

    L += ["## Decisions", ""]
    decisions = state.get("decisions", [])
    L += [f"{i}. {d['action']} {d.get('columns') or ''} [{d.get('source')}]: {d.get('reason')}"
          for i, d in enumerate(decisions, 1)] or ["- none"]
    L.append("")

    L += ["## Models compared (validation)", "",
          "| Model | ROC-AUC | F1 | Balanced acc. | Rare-class recall |", "|---|---|---|---|---|"]
    for name, r in state.get("results", {}).items():
        L.append(f"| {name} | {_fmt(r.get('val_roc_auc'))} | {_fmt(r.get('f1'))} | "
                 f"{_fmt(r.get('balanced_accuracy'))} | {_fmt(r.get('rare_class_recall'))} |")
    L.append("")

    if state.get("test_score") is not None:
        L += [f"**Final score on the locked test set (ROC-AUC): {_fmt(state['test_score'])}**", ""]

    L += ["## Critic", ""]
    v = state.get("critic_verdict")
    if v:
        L.append(f"Verdict: {v.get('verdict')} ({v.get('source')}): {v.get('reason')}")
    else:
        L.append("The critic did not run for this system.")
    L += [f"- {f['check']} [{f['severity']}]: {f['details']}"
          for f in state.get("critic_flags", [])] + [""]

    e = state.get("explanation")
    if e:
        L += ["## Explanation", "", e["text"], "",
              "Top features: " + ", ".join(f"{t['feature']} ({t['share_percent']}%)"
                                           for t in e["top_features"]), ""]
    return "\n".join(L)


def run_reporter(state: dict) -> dict:
    return {**state, "report": build_report(state)}