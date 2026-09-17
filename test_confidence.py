# Turn logprobs into a clean "probability of Yes"
import math
import ollama

def yes_probability(question):
    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": question + " Answer with one word: Yes or No."}],
        logprobs=True,
        top_logprobs=20,
        options={"temperature": 0, "num_predict": 1},
    )
    yes = 0.0
    no = 0.0
    # Look at the 20 most likely first tokens
    for candidate in response.logprobs[0].top_logprobs:
        word = candidate.token.strip().lower()      # ' Yes', 'YES' -> 'yes'
        prob = math.exp(candidate.logprob)          # log-probability -> probability
        if word == "yes":
            yes += prob
        elif word == "no":
            no += prob
    return yes / (yes + no)                         # share of Yes out of Yes+No

questions = [
    "Is the sky blue on a clear day?",                                  # easy
    "Is 2 + 2 equal to 5?",                                             # easy, answer No
    "A student wrote: 'Plants get their food from the soil.' Is this answer fully correct?",  # tricky
    "Is a tomato a vegetable?",                                         # ambiguous
]

for q in questions:
    p = yes_probability(q)
    print(f"P(Yes) = {p:.3f}   | {q}")