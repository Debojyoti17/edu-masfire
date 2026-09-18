# Functions for talking to local Ollama models.
# Two things live here: getting plain text back, and getting the model's
# confidence in a multiple-choice answer (read from logprobs).

import math
import ollama

DEFAULT_MODEL = "qwen2.5:7b"

# Options we always pass to Ollama so every call is deterministic and repeatable.
BASE_OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 4096}


def generate(prompt, model=DEFAULT_MODEL, max_tokens=None, temperature=None, seed=None):
    """Send a prompt to an Ollama model and return its text reply as a string.

    max_tokens: if given, caps how many tokens the model is allowed to
    generate (passed to Ollama as num_predict). Left unset means no cap.

    temperature, seed: if given, override the project defaults (0 and 42)
    for this call only. Needed for things like self-consistency sampling,
    which requires the model to actually vary its answer across repeated
    calls - impossible under the usual fixed temperature=0/seed=42.
    """
    options = dict(BASE_OPTIONS)
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    if temperature is not None:
        options["temperature"] = temperature
    if seed is not None:
        options["seed"] = seed

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options=options,
    )
    return response.message.content


def choice_probabilities(prompt, options, model=DEFAULT_MODEL, top_logprobs=20, prefill=None):
    """
    Ask a multiple-choice question (e.g. prompt tells the model to answer
    with one letter like "A") and return how confident the model was in
    each option, read from the logprobs of the first generated token.

    options: a list of short strings the model was told to answer with,
             e.g. ["A", "B", "C", "D"].

    prefill: optional text to plant as the start of the model's reply
             (added as a final assistant message), so the model continues
             from that text instead of starting a fresh reply. Ending it
             with something like "Label: " forces the very next token to
             be the answer, instead of the model wanting to open with a
             whole sentence - which otherwise leaves most of the
             probability mass on non-option tokens and makes "mass" low.

    Returns a dict with two parts:
      "probs": dict of option -> probability, renormalized so the values
               sum to 1. This is None if no option was found at all among
               the top candidate tokens (can't divide by zero mass).
      "mass":  the raw probability (0 to 1) that the top candidate tokens
               assigned to these options *before* renormalizing. Low mass
               means the model mostly wanted to say something else, which
               is a useful warning sign (e.g. a corrupted prompt).
    """
    messages = [{"role": "user", "content": prompt}]
    if prefill is not None:
        messages.append({"role": "assistant", "content": prefill})

    response = ollama.chat(
        model=model,
        messages=messages,
        logprobs=True,
        top_logprobs=top_logprobs,
        options={**BASE_OPTIONS, "num_predict": 1},  # only need the first token
    )

    # A token like " A", "A" or "a" should all count as the option "A".
    # Build a lookup from the normalized (stripped, lowercased) form of each
    # option back to the option's original spelling, so we can match candidate
    # tokens against it regardless of case or leading spaces.
    normalized_to_option = {option.strip().lower(): option for option in options}

    # Raw probability mass collected for each option so far.
    raw = {option: 0.0 for option in options}

    for candidate in response.logprobs[0].top_logprobs:
        normalized_token = candidate.token.strip().lower()
        if normalized_token in normalized_to_option:
            option = normalized_to_option[normalized_token]
            raw[option] += math.exp(candidate.logprob)  # logprob -> real probability

    mass = sum(raw.values())

    if mass == 0:
        probs = None
    else:
        probs = {option: value / mass for option, value in raw.items()}

    return {"probs": probs, "mass": mass}
