from typing import TypedDict


class State(TypedDict, total=False):
    dataset_name: str
    goal: str
    target: str
    train_path: str
    val_path: str
    profile: dict
    features: list
    dropped: dict        # column -> reason
    results: dict        # model -> validation scores
    best_model: str
    critic_flags: list
    loop_count: int
    next_step: str
    report: str