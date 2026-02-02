# llama_infer.py
import typing as _t
from llama_cpp import Llama


class LlamaInference:
    def __init__(
        self,
        model_path: str,
        lora_path: _t.Optional[str] = None,
        n_ctx: int = 2048,
    ):
        """
        Wrapper around llama_cpp.Llama.

        Args:
            model_path: Path to base GGUF model.
            lora_path:  Optional path to LoRA GGUF adapter. If provided,
                       llama.cpp will load the adapter on top of the base.
            n_ctx:      Context length.
        """
        kwargs = dict(
            model_path=model_path,
            n_ctx=n_ctx,
            verbose=False,
            logits_all=True,
        )

        # If a LoRA adapter is provided, pass it to llama_cpp
        if lora_path is not None and lora_path != "":
            kwargs["lora_path"] = lora_path

        self.model = Llama(**kwargs)

    def infer(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.1,
        stop: _t.Optional[_t.List[str]] = None,
        logprobs_k: _t.Optional[int] = None,  # ask for top-k logprobs if provided
    ):
        """
        Run inference and optionally return top-k logprobs.

        Returns:
            If logprobs_k is not None and logprobs are available:
                (text, token_logprobs, top_logprobs)
            Else:
                (text, token_logprobs)   # token_logprobs may be [] if not available
        """
        call_kwargs = dict(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        if logprobs_k is not None:
            call_kwargs["logprobs"] = int(logprobs_k)

        result = self.model(**call_kwargs)
        c0 = result["choices"][0]
        text = c0["text"].strip()

        logprobs_obj = c0.get("logprobs")  # may be None if logprobs not requested

        # Case 1: logprobs requested and present → return text, token_logprobs, top_logprobs
        if logprobs_k is not None and logprobs_obj is not None:
            token_logprobs = logprobs_obj.get("token_logprobs") or []
            top_logprobs = logprobs_obj.get("top_logprobs") or []
            return text, token_logprobs, top_logprobs

        # Case 2: logprobs not requested or missing → just return text + empty list
        token_logprobs = []
        return text, token_logprobs
