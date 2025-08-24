# llama_infer.py
from llama_cpp import Llama

class LlamaInference:
    def __init__(self, model_path, n_ctx=2048):
        # Suppress all logs with verbose=False
        self.model = Llama(model_path=model_path, n_ctx=n_ctx, verbose=False, logits_all=True)

    def infer(self, prompt, max_tokens=256, temperature=0.1, stop=None, logprobs=1):
        """
        Generate a response and return both text and token logprobs for confidence.
        """
        result = self.model(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            logprobs=logprobs
        )
        response = result["choices"][0]["text"].strip()
        token_logprobs = result["choices"][0]["logprobs"]["token_logprobs"]
        return response, token_logprobs
