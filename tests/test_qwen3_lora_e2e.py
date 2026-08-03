"""Single Qwen3 dense TP2/DP2 native-LoRA rollout/train/PEFT E2E."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from areno.adapters import LoraConfig
from areno.api import ArenoConfig, Trainer
from areno.api.algorithms import get_algorithm
from areno.api.trainer_config import PolicyTrainerConfig
from areno.api.trainers.policy_only import PolicyOnlyTrainer


class _ObservedTrainer:
    def __init__(self, inner: Trainer) -> None:
        self.inner = inner
        self.rollout_versions: list[int | None] = []
        self.train_versions: list[int | None] = []

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    async def rollout_token_batch_async(self, prompt_tokens, n_samples, sampling_params):
        results = await self.inner.rollout_token_batch_async(prompt_tokens, n_samples, sampling_params)
        self.rollout_versions.extend(result.adapter_version for result in results)
        return results

    def train(self, batch_data, loss_fn, mini_bs=8, gradient_accumulation_steps=None):
        result = self.inner.train(batch_data, loss_fn, mini_bs, gradient_accumulation_steps)
        self.train_versions.append(result.get("adapter_version"))
        return result


def test_qwen3_dense_lora_tp2_dp2_rollout_train_peft(tmp_path: Path) -> None:
    model_path_value = os.getenv("ARENO_E2E_QWEN3_MODEL")
    if not model_path_value:
        pytest.skip("set ARENO_E2E_QWEN3_MODEL to run the 4-GPU LoRA E2E")
    model_path = Path(model_path_value)
    initial_path = tmp_path / "adapter-initial"
    final_path = tmp_path / "adapter-final"
    lora = LoraConfig()
    backend_config = ArenoConfig(
        tp_size=2,
        dp_size=2,
        devices=[0, 1, 2, 3],
        lora=lora,
        max_running_prompts=4,
        optimizer={
            "lr": 1.0e-3,
            "min_lr": 1.0e-3,
            "lr_decay_style": "constant",
            "weight_decay": 0.0,
            "grad_clip_norm": 1.0,
        },
        runtime={
            "compile_model": False,
            "activation_checkpointing": False,
            "keep_rollout_state": False,
        },
    )
    inner = Trainer(4, os.fspath(model_path), custom_config=backend_config)
    observed = _ObservedTrainer(inner)
    config = PolicyTrainerConfig(
        algo="grpo",
        ckpt=os.fspath(model_path),
        dataset_path="e2e://in-memory",
        epochs=1,
        max_steps=2,
        world_size=4,
        tp_size=2,
        train_devices=[0, 1, 2, 3],
        batch_size=1,
        mini_bs=4,
        n_samples=4,
        max_running_prompts=4,
        max_prompt_tokens=64,
        max_new_tokens=16,
        optimizer_lr=1.0e-3,
        optimizer_min_lr=1.0e-3,
        lr_decay_style="constant",
        weight_decay=0.0,
        activation_checkpointing=False,
        keep_rollout_state=False,
        metrics_log_dir=None,
        lora=lora,
    )
    dataset = [
        {"prompt": "Write one uncommon English noun. Output only the noun."},
        {"prompt": "Invent one short fictional name. Output only the name."},
    ]
    def reward_fn(record) -> float:
        return float(record.metadata["sample_index"])
    policy = PolicyOnlyTrainer(
        config,
        instance=observed,
        dataset=dataset,
        reward_fn=reward_fn,
        loss_fn=get_algorithm("grpo").make_loss_fn(config),
    )

    observed.init()
    try:
        observed.export_adapter(os.fspath(initial_path))
        policy._fit_initialized()
        replica_max_diff = inner._backend._require_train_engine().adapter_replica_max_diff()
        observed.export_adapter(os.fspath(final_path))
    finally:
        observed.close()

    assert observed.rollout_versions == [0, 1]
    assert observed.train_versions == [1, 2]
    assert replica_max_diff <= 1.0e-6
    initial = load_file(initial_path / "adapter_model.safetensors")
    final = load_file(final_path / "adapter_model.safetensors")
    assert any(not torch.equal(initial[name], final[name]) for name in initial)

    token_ids, peft_logprobs = _peft_logprobs(model_path, final_path)
    imported = Trainer(
        2,
        os.fspath(model_path),
        custom_config=ArenoConfig(
            tp_size=2,
            dp_size=1,
            devices=[0, 1],
            lora=LoraConfig(adapter_path=os.fspath(final_path)),
            runtime={"compile_model": False, "activation_checkpointing": False},
        ),
    )
    imported.init()
    try:
        areno_logprobs = imported.score_logprobs("actor", [token_ids], microbatch_size=1)[0]
    finally:
        imported.close()
    torch.testing.assert_close(
        torch.tensor(areno_logprobs[1:]),
        torch.tensor(peft_logprobs),
        rtol=3.0e-2,
        atol=3.0e-2,
    )


def _peft_logprobs(model_path: Path, adapter_path: Path) -> tuple[list[int], list[float]]:
    peft_source = os.getenv("ARENO_E2E_PEFT_SOURCE")
    if peft_source:
        sys.path.insert(0, peft_source)
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    token_ids = tokenizer.encode("A short adapter parity check.", add_special_tokens=True)
    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to("cuda:0")
    model = PeftModel.from_pretrained(base, adapter_path, autocast_adapter_dtype=False).eval()
    tokens = torch.tensor([token_ids], device="cuda:0", dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids=tokens).logits[0, :-1].float()
        selected = logits.log_softmax(dim=-1).gather(-1, tokens[0, 1:].unsqueeze(-1)).squeeze(-1)
    result = selected.cpu().tolist()
    del model, base, tokens, logits, selected
    torch.cuda.empty_cache()
    return token_ids, result
