# Reading handwritten answer sheets: a local vision model (through Ollama) turns a photo of a
# student's handwritten answer into text, so it can be graded like a typed answer.
# Also a simulated "misread handwriting" fault: text with the kind of mistakes handwriting reading
# makes (similar-looking letters swapped, letters dropped, words run together), used to test the
# pipeline on every test set without needing thousands of photos.
import hashlib
import io
import json
import math
import os
import random
import re

from PIL import Image, ImageOps

from src.llm import BASE_OPTIONS, chat

VISION_MODEL = "qwen2.5vl:7b"
READINGS_FILE = os.path.join("results", "pipeline_v2", "cache", "photos.jsonl")
_saved = {}
READ_PROMPT = """This photo shows a student's handwritten answer to a school science question.
Write out exactly what the student wrote, word for word, keeping their spelling mistakes.
Leave out the code at the top of the page (like H01 or P01). Reply with the answer text only."""
# Pairs of letters (or letter groups) that handwriting readers often confuse.
LOOKALIKES = [("rn", "m"), ("m", "rn"), ("cl", "d"), ("d", "cl"), ("l", "1"), ("i", "l"), ("o", "a"), ("a", "o"),
              ("e", "c"), ("c", "e"), ("u", "v"), ("v", "u"), ("n", "h"), ("h", "n"), ("t", "f"), ("g", "q"), ("s", "5")]


def read_image(image, model=VISION_MODEL):
    """Read one (already rotated) image. Returns (text, reading confidence), where the confidence is
    the average probability the model gave the words it wrote (None if unavailable)."""
    data = io.BytesIO()
    image.convert("RGB").save(data, format="JPEG", quality=95)
    response = chat(model=model, messages=[{"role": "user", "content": READ_PROMPT, "images": [data.getvalue()]}],
                    logprobs=True, options={**BASE_OPTIONS, "num_predict": 200})
    text = re.sub(r"^\s*[HP]\d{1,3}\b[\s:.\-]*", "", response.message.content.strip())  # H01 real, P01 example sheets
    probs = [math.exp(t.logprob) for t in (response.logprobs or [])]
    return text.strip(), (sum(probs) / len(probs) if probs else None)


def read_photo(path, model=VISION_MODEL, rotations=(0, 90, 180, 270)):
    """Read one photo of a handwritten answer. The photo is first turned the way the phone says it
    was held (its orientation tag), then read once in each rotation, and the reading the model is
    most confident about is kept - a photo taken sideways is otherwise read much worse. The true
    text is never used to choose. Returns (text, reading confidence)."""
    with open(path, "rb") as f:
        key = hashlib.sha1(f.read()).hexdigest() + f"|{model}|{rotations}"
    saved = saved_readings()
    if key not in saved:
        image = ImageOps.exif_transpose(Image.open(path))
        readings = [read_image(image.rotate(angle, expand=True), model) for angle in rotations]
        saved[key] = max(readings, key=lambda tc: tc[1] or 0)
        with open(READINGS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "value": saved[key]}) + "\n")
    return tuple(saved[key])


def saved_readings():
    """Readings of photos read before (same file content, model and rotations), so re-grading a
    class does not read every photo again."""
    if "readings" not in _saved:
        _saved["readings"] = {}
        if os.path.exists(READINGS_FILE):
            with open(READINGS_FILE, encoding="utf-8") as f:
                _saved["readings"] = {r["key"]: r["value"] for r in map(json.loads, f)}
    return _saved["readings"]


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
