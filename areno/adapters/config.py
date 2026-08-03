"""Public configuration for the Qwen3 dense LoRA runtime."""

from __future__ import annotations

from dataclasses import dataclass

QWEN3_DENSE_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True, slots=True)
class LoraConfig:
    """Supported PEFT-compatible LoRA subset for dense Qwen3 models."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = QWEN3_DENSE_TARGETS
    adapter_path: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_modules", tuple(self.target_modules))
        if self.rank < 1:
            raise ValueError("lora rank must be >= 1")
        if self.alpha <= 0:
            raise ValueError("lora alpha must be > 0")
        if self.dropout != 0.0:
            raise ValueError("native Qwen3 LoRA currently requires dropout=0")
        requested = set(self.target_modules)
        supported = set(QWEN3_DENSE_TARGETS)
        if not requested or not requested <= supported:
            raise ValueError(f"target_modules must be a non-empty subset of {QWEN3_DENSE_TARGETS}")

    @property
    def scale(self) -> float:
        return float(self.alpha) / float(self.rank)
