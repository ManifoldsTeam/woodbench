"""Optional Weights & Biases logging for training runs."""

from __future__ import annotations

from typing import Any, Dict, Optional


class WandbTracker:
    """Thin wrapper around wandb.init / log / finish."""

    def __init__(self, run: Any) -> None:
        self._run = run

    @classmethod
    def start(
        cls,
        project: str,
        name: str,
        config: Dict[str, Any],
        entity: Optional[str] = None,
        tags: Optional[list[str]] = None,
        dir_path: Optional[str] = None,
    ) -> Optional["WandbTracker"]:
        try:
            import wandb
        except ImportError:
            return None

        init_kwargs: Dict[str, Any] = {
            "project": project,
            "name": name,
            "config": config,
            "reinit": True,
        }
        if entity:
            init_kwargs["entity"] = entity
        if tags:
            init_kwargs["tags"] = tags
        if dir_path:
            init_kwargs["dir"] = dir_path

        run = wandb.init(**init_kwargs)
        return cls(run)

    def log(self, metrics: Dict[str, float], step: int) -> None:
        if self._run is not None:
            self._run.log(metrics, step=step)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
