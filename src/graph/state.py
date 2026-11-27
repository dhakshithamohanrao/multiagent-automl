from typing import TypedDict


class State(TypedDict, total=False):
    dataset_name: str
    goal: str
    target: str
    seed: int
    train_path: str
    val_path: str
    profile: dict
    features: list
    dropped: dict        # column -> reason
    balance: bool        # set by the lead's fix_imbalance action (Phase 5)
    models: dict         # model name -> fitted pipeline (in memory only, never logged)
    tuned_params: dict   # model name -> {"params": ..., "cv_auc": ...}
    results: dict        # model -> validation scores
    best_model: str
    critic_flags: list
    loop_count: int
    decisions: list      # lead's decisions: action, columns, reason, source
    next_step: str
    report: str