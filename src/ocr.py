# Reading handwritten answer sheets: a local vision model (through Ollama) turns a photo of a
# student's handwritten answer into text, so it can be graded like a typed answer.
# Also a simulated "misread handwriting" fault: text with the kind of mistakes handwriting reading
# makes (similar-looking letters swapped, letters dropped, words run together), used to test the
# pipeline on every test set without needing thousands of photos.
import math
import random
import re

from src.llm import BASE_OPTIONS, chat

VISION_MODEL = "qwen2.5vl:7b"
READ_PROMPT = """This photo shows a student's handwritten answer to a school science question.
Write out exactly what the student wrote, word for word, keeping their spelling mistakes.
Leave out the code at the top of the page (like H01 or P01). Reply with the answer text only."""
# Pairs of letters (or letter groups) that handwriting readers often confuse.
LOOKALIKES = [("rn", "m"), ("m", "rn"), ("cl", "d"), ("d", "cl"), ("l", "1"), ("i", "l"), ("o", "a"), ("a", "o"),
              ("e", "c"), ("c", "e"), ("u", "v"), ("v", "u"), ("n", "h"), ("h", "n"), ("t", "f"), ("g", "q"), ("s", "5")]


def read_photo(path, model=VISION_MODEL):
    """Read one photo of a handwritten answer. Returns (text, reading confidence), where the
    confidence is the average probability the model gave the words it wrote (None if unavailable)."""
    response = chat(model=model, messages=[{"role": "user", "content": READ_PROMPT, "images": [path]}],
                    logprobs=True, options={**BASE_OPTIONS, "num_predict": 200})
    text = re.sub(r"^\s*[HP]\d{1,3}\b[\s:.\-]*", "", response.message.content.strip())  # H01 real, P01 example sheets
    probs = [math.exp(t.logprob) for t in (response.logprobs or [])]
    return text.strip(), (sum(probs) / len(probs) if probs else None)


def simulate_misreading(text, seed, rate=0.12):
    """The text as a handwriting reader might misread it: each letter is swapped for a look-alike or
    dropped with probability `rate`, and occasionally two words are run together. The same seed always
    gives the same result."""
    rng = random.Random(seed)
    out, i = [], 0
    while i < len(text):
        if text[i].isalpha() and rng.random() < rate:
            for a, b in LOOKALIKES:
                if text.startswith(a, i) and rng.random() < 0.5:
                    out.append(b)
                    i += len(a)
                    break
            else:
                if rng.random() < 0.3:
                    i += 1  # letter dropped
                else:
                    out.append(text[i])
                    i += 1
            continue
        if text[i] == " " and rng.random() < rate / 3:
            i += 1  # two words run together
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def character_error_rate(truth, reading):
    """Share of characters that must be changed, added or removed to turn the reading into the
    true text (edit distance divided by the true text's length; 0 = perfect)."""
    a, b = truth.strip(), reading.strip()
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(1, len(a))
