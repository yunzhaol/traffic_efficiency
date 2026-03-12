"""utils/logger.py — Lightweight CSV logger for training metrics."""
import csv
import os
from typing import Dict, Any, List


class Logger:
    def __init__(self):
        self._records: List[Dict[str, Any]] = []
        self._fieldnames: List[str] = []

    def log(self, metrics: Dict[str, Any]):
        self._records.append(metrics)
        for k in metrics:
            if k not in self._fieldnames:
                self._fieldnames.append(k)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._records)
        print(f"[Logger] Saved {len(self._records)} records → {path}")

    def last(self, key: str, default=None):
        if self._records:
            return self._records[-1].get(key, default)
        return default
