"""Temporal model evidence; time near a door does not identify staff."""
from dataclasses import dataclass
import math


@dataclass
class StaffEvidence:
    samples: int = 0
    positive: int = 0
    confirmed: bool = False

    def observe(self, probability, threshold, required):
        if probability is None:
            return
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Staff probability must be finite and in [0,1]")
        self.samples += 1
        self.positive = self.positive + 1 if probability >= threshold else 0
        if self.positive >= required:
            self.confirmed = True

    def ready(self, required):
        return self.confirmed or self.samples >= required
