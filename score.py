import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Sentences
sentences = [
    "Employee must maintain confidentiality",
    "Staff should not share company data",
    "Company offers medical insurance"
]

# Get embeddings (lightweight + fast model)
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=sentences
)

# Extract vectors
embeddings = [item.embedding for item in response.data]

# Cosine similarity function
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Compute similarity matrix
print("\n Semantic Similarity:\n")

for i in range(len(sentences)):
    for j in range(len(sentences)):
        score = cosine_similarity(embeddings[i], embeddings[j])
        print(f"'{sentences[i]}'")
        print(f"vs")
        print(f"'{sentences[j]}'")
        print(f"→ Similarity: {score:.3f}\n")