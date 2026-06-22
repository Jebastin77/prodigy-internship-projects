"""
Markov Chain Text Generator
Core statistical model for text generation.
"""

import random
import re
from collections import defaultdict


class MarkovChain:
    def __init__(self, order: int = 2):
        self.order = order          # n-gram size
        self.model: dict = defaultdict(list)
        self.starters: list = []    # sentence-starting n-grams

    # ── Build ──────────────────────────────────────────────────────────────
    def train(self, text: str) -> None:
        """Build transition table from raw text."""
        words = re.findall(r"\S+", text)
        if len(words) <= self.order:
            return

        for i in range(len(words) - self.order):
            key = tuple(words[i: i + self.order])
            nxt = words[i + self.order]
            self.model[key].append(nxt)
            if i == 0 or words[i - 1].endswith((".", "!", "?")):
                self.starters.append(key)

    # ── Generate ───────────────────────────────────────────────────────────
    def generate(self, length: int = 50, seed: str | None = None) -> str:
        """Generate `length` words of text."""
        if not self.model:
            return "Model not trained yet."

        # Pick start key
        if seed:
            seed_words = seed.strip().split()
            if len(seed_words) >= self.order:
                key = tuple(seed_words[-self.order:])
                if key not in self.model:
                    key = random.choice(self.starters)
            else:
                key = random.choice(self.starters)
        else:
            key = random.choice(self.starters) if self.starters else random.choice(list(self.model))

        result = list(key)
        for _ in range(length - self.order):
            choices = self.model.get(key)
            if not choices:
                break
            nxt = random.choice(choices)
            result.append(nxt)
            key = tuple(result[-self.order:])

        return " ".join(result)

    # ── Stats ──────────────────────────────────────────────────────────────
    def stats(self) -> dict:
        return {
            "states": len(self.model),
            "transitions": sum(len(v) for v in self.model.values()),
            "order": self.order,
        }
