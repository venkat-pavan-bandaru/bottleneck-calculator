# logger.py
# Handles all console output so print() is never scattered across files.
# OOP concept: Encapsulation — formatting logic is hidden inside the class.

from datetime import datetime

class Logger:

    def __init__(self, name: str):
        self.name = name          # e.g. "DataLoader", "LinearRegression"

    def _timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def info(self, msg: str):
        print(f"[{self._timestamp()}]  {self.name:<20} {msg}")

    def success(self, msg: str):
        print(f"[{self._timestamp()}]  {self.name:<20} ✓  {msg}")

    def section(self, title: str):
        print(f"\n{'─' * 55}")
        print(f"  {title}")
        print(f"{'─' * 55}")

    def cost(self, iteration: int, cost_value: float):
        # Special formatter for the training loop
        print(f"  iter {iteration:>5}   cost = {cost_value:.6f}")
