"""Lazy Hugging Face Transformers backend with optional CPU/CUDA optimizations."""

from __future__ import annotations

import importlib.util
import time
from contextlib import nullcontext
from typing import Any

from budgetroute.config import BackendConfig
from budgetroute.exceptions import BackendError
from budgetroute.schemas import BackendGeneration, BackendName, GenerationRequest


class TransformersBackend:
    def __init__(self, name: BackendName, config: BackendConfig) -> None:
        self.name = name
        self.config = config
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._device = config.device
        self._optimization: dict[str, Any] = {
            "compile_requested": config.compile,
            "compile_enabled": False,
            "quantization_requested": config.quantization,
            "quantization_enabled": False,
        }

    def _require_dependencies(self) -> None:
        missing = [
            name for name in ("torch", "transformers") if importlib.util.find_spec(name) is None
        ]
        if missing:
            raise BackendError(
                "Transformers backend requires optional dependencies: "
                + ", ".join(missing)
                + ". Install with `pip install -e .[transformers]`."
            )

    def initialize(self) -> None:
        if self._model is not None:
            return
        self._require_dependencies()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        if self.config.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self.config.device == "cuda" and not torch.cuda.is_available():
            raise BackendError(
                "CUDA was explicitly requested but torch.cuda.is_available() is false. "
                "Install a CUDA-enabled PyTorch build or select device=cpu."
            )
        dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.config.precision]
        if self._device == "cuda" and self.config.precision == "bf16":
            supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
            if not supported:
                raise BackendError(
                    "BF16 was requested but the selected CUDA device does not support it"
                )
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": dtype,
        }
        if self.config.quantization != "none":
            if importlib.util.find_spec("bitsandbytes") is None:
                raise BackendError(
                    "quantization requires bitsandbytes; install the gpu extra on a supported platform"
                )
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=self.config.quantization == "int8",
                load_in_4bit=self.config.quantization == "int4",
            )
            self._optimization["quantization_enabled"] = True
        model_id = self.config.model_id
        if model_id is None:
            raise BackendError("model_id is required for a Transformers backend")
        tokenizer_id = self.config.tokenizer_id or model_id
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_id, trust_remote_code=self.config.trust_remote_code
            )
            self._model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        except Exception as exc:
            self.cleanup()
            raise BackendError(
                f"could not load Transformers model {model_id!r}: {type(exc).__name__}: {exc}"
            ) from exc
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        if self.config.quantization == "none":
            self._model.to(self._device)
        self._model.eval()
        if self.config.compile:
            if not hasattr(torch, "compile"):
                raise BackendError(
                    "torch.compile was requested but is unavailable in this PyTorch build"
                )
            self._model = torch.compile(self._model)
            self._optimization["compile_enabled"] = True

    def health(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "type": "transformers",
            "initialized": self._model is not None,
            "ready": self._model is not None,
            "device": self._device,
        }

    def token_count(self, text: str) -> int:
        if self._tokenizer is None:
            raise BackendError("backend must be initialized before token counting")
        return len(self._tokenizer.encode(text, add_special_tokens=True))

    def _render_prompt(self, prompt: str) -> str:
        if self._tokenizer is None:
            raise BackendError("backend is not initialized")
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            return str(
                self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return prompt

    def generate(self, request: GenerationRequest) -> BackendGeneration:
        if self._model is None or self._tokenizer is None or self._torch is None:
            raise BackendError("Transformers backend is not initialized")
        torch = self._torch
        prompt = self._render_prompt(request.prompt)
        encoded = self._tokenizer(prompt, return_tensors="pt", padding=True)
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        input_length = int(encoded["input_ids"].shape[-1])
        generation = self.config.generation
        max_new_tokens = request.max_new_tokens or generation.max_new_tokens
        synchronize = self._device == "cuda"
        if synchronize:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter_ns()
        autocast = (
            torch.autocast(
                device_type="cuda",
                dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[self.config.precision],
            )
            if self._device == "cuda" and self.config.precision in {"fp16", "bf16"}
            else nullcontext()
        )
        try:
            with torch.inference_mode(), autocast:
                output = self._model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=generation.temperature > 0,
                    temperature=max(generation.temperature, 1e-6),
                    top_p=generation.top_p,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
        except Exception as exc:
            raise BackendError(
                f"Transformers generation failed: {type(exc).__name__}: {exc}"
            ) from exc
        if synchronize:
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        generated_ids = output[0, input_length:]
        text = str(self._tokenizer.decode(generated_ids, skip_special_tokens=True)).strip()
        output_tokens = int(generated_ids.shape[-1])
        confidence = 0.5
        return BackendGeneration(
            text=text,
            backend=self.name,
            input_tokens=input_length,
            output_tokens=output_tokens,
            confidence=confidence,
            generation_ms=elapsed_ms,
            metadata={
                "fake": False,
                "device": self._device,
                "tokens_per_second": output_tokens / (elapsed_ms / 1000) if elapsed_ms else None,
                "peak_cuda_mb": (
                    torch.cuda.max_memory_allocated() / (1024 * 1024) if synchronize else None
                ),
            },
        )

    def generate_batch(self, requests: list[GenerationRequest]) -> list[BackendGeneration]:
        # A stable direct implementation; optimized padding-aware batching can replace this
        # without changing the protocol or the API microbatcher.
        return [self.generate(request) for request in requests]

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "type": "transformers",
            "model_id": self.config.model_id,
            "tokenizer_id": self.config.tokenizer_id or self.config.model_id,
            "device_requested": self.config.device,
            "device_resolved": self._device,
            "precision": self.config.precision,
            **self._optimization,
        }

    def cleanup(self) -> None:
        self._model = None
        self._tokenizer = None
        if self._torch is not None and getattr(self._torch.cuda, "is_available", lambda: False)():
            self._torch.cuda.empty_cache()
