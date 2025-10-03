from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

text = "hello world"
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=text
)

embedding = response.data[0].embedding
print(f"Embedding length: {len(embedding)}")
print(f"First 10 dims: {embedding[:10]}")
