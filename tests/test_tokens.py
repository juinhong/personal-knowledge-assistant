import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o-mini")

long_text = "Your name is Claude and you are a helpful assistant. " * 50

tokens = encoder.encode(long_text)
token_count = len(tokens)

cost_per_million = 0.15  # gpt-4o-mini input cost in USD
cost = (token_count / 1_000_000) * cost_per_million

print(f"Token count: {token_count}")
print(f"Estimated cost: ${cost:.6f}")