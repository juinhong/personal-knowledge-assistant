from dotenv import load_dotenv
from openai import OpenAI
import math

load_dotenv()
client = OpenAI()

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x ** 2 for x in a))
    mag_b = math.sqrt(sum(x ** 2 for x in b))
    return dot / (mag_a * mag_b)

# Embed some sentences
sentences = [
    "how do I fix a bug?",
    "debugging techniques for developers",
    "I love eating pizza",
    "my favourite food is pasta",
]

embeddings = [get_embedding(s) for s in sentences]

print(f"Each embedding has {len(embeddings[0])} dimensions\n")

# Compare all pairs
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        score = cosine_similarity(embeddings[i], embeddings[j])
        print(f"'{sentences[i]}'\n'{sentences[j]}'\nSimilarity: {score:.4f}\n")