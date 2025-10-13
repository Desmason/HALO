import math
import random

# =========================
# --- CONFIDENCE MATH -----
# =========================

def compute_confidence(token_logprobs):
    """
    Convert llama.cpp token log-probs into a [0,1] confidence and return perplexity.
    Based on perplexity derived from average log-prob.
    """
    if not token_logprobs:
        return 0.0, None

    vals = [lp for lp in token_logprobs if lp is not None]
    if not vals:
        return 0.0, None

    avg_logprob = sum(vals) / len(vals)
    ppl = math.exp(-avg_logprob)  # perplexity = e^{-avg_logprob}
    conf = 1.0 / (1.0 + ppl)
    conf = float(max(0.0, min(1.0, conf)))
    return conf, ppl


def _entropy_from_toplogprobs(top_logprobs_step):
    if not top_logprobs_step:
        return None
    ps = [math.exp(lp) for lp in top_logprobs_step.values()]
    Z = sum(ps)
    if Z <= 0:
        return None
    ps = [p / Z for p in ps]
    return -sum(p * math.log(p + 1e-12) for p in ps)


def compute_confidence_entropy(top_logprobs_list, max_entropy=None):
    """Average token-level entropy → confidence in [0,1], also return avg entropy."""
    entropies = []
    for step in top_logprobs_list:
        h = _entropy_from_toplogprobs(step)
        if h is not None:
            entropies.append(h)

    if not entropies:
        return 0.0, None

    avg_H = sum(entropies) / len(entropies)
    if max_entropy is None:
        max_entropy = math.log(5.0)
    conf = 1.0 - min(avg_H, max_entropy) / max_entropy
    conf = float(max(0.0, min(1.0, conf)))
    return conf, avg_H


def compute_hybrid_confidence(token_logprobs, top_logprobs):
    """Simple average of perplexity- and entropy-based confidences."""
    conf_ppl, _ = compute_confidence(token_logprobs)
    conf_ent, _ = compute_confidence_entropy(top_logprobs)
    return 0.5 * conf_ppl + 0.5 * conf_ent


# =========================
# --- ROUTING MATH --------
# =========================

def interp_threshold(phi: float, tau_local: float = 0.45, tau_cloud: float = 0.75) -> float:
    """τ'(φ) = (1-φ) τ_cloud + φ τ_local"""
    return (1.0 - phi) * tau_cloud + phi * tau_local


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def probability_local(c: float, phi: float, s: float = 20.0,
                      tau_local: float = 0.50, tau_cloud: float = 0.80) -> float:
    tau_prime = interp_threshold(phi, tau_local, tau_cloud)
    return sigmoid(s * (c - tau_prime))


def decide_route(c: float, phi: float, deterministic: bool = True,
                 s: float = 20.0, tau_local: float = 0.45, tau_cloud: float = 0.75) -> str:
    p_local = probability_local(c, phi, s=s, tau_local=tau_local, tau_cloud=tau_cloud)
    if deterministic:
        return "local" if p_local >= 0.6 else "openai"
    else:
        return "local" if random.random() < p_local else "openai"
