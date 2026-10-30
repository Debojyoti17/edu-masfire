# Test: can we get token probabilities from Ollama?
import ollama

response = ollama.chat(
    model="qwen2.5:7b",
    messages=[{"role": "user",
               "content": "Is the sky blue on a clear day? Answer with one word: Yes or No."}],
    logprobs=True,          # ask for the probability of each token
    top_logprobs=5,         # also show the 5 most likely alternatives
    options={"temperature": 0, "num_predict": 1},   # deterministic, 1 token only
)

print("ANSWER:", response.message.content)
print("FULL RESPONSE:")
print(response)