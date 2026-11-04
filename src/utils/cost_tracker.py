import time


class CostTracker:
    """Measures one run: time, LLM calls, tokens and estimated cost."""

    def __init__(self):
        self._start = time.perf_counter()
        self.llm_calls = 0
        self.tokens = 0
        self.cost = 0.0

    def add_llm_call(self, tokens: int = 0, cost: float = 0.0) -> None:
        self.llm_calls += 1
        self.tokens += tokens
        self.cost += cost

    def summary(self) -> dict:
        return {
            "time_s": round(time.perf_counter() - self._start, 2),
            "llm_calls": self.llm_calls,
            "tokens": self.tokens,
            "cost": round(self.cost, 4),
        }