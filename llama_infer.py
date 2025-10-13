# llama_infer.py
import typing as _t
from llama_cpp import Llama

class LlamaInference:
    def __init__(self, model_path, n_ctx=2048):
        self.model = Llama(model_path=model_path, n_ctx=n_ctx, verbose=False, logits_all=True)

    def infer(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.1,
        stop: _t.Optional[_t.List[str]] = None,
        logprobs_k: _t.Optional[int] = None,   # <-- new: ask for top-k logprobs if provided
    ):
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
        token_logprobs = c0["logprobs"]["token_logprobs"]  # list[float|None]

        if logprobs_k is not None:
            top_logprobs = c0["logprobs"]["top_logprobs"]  # list[dict[token]->logprob]
            return text, token_logprobs, top_logprobs

        return text, token_logprobs
