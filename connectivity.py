from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_UR"))

emb = client.embeddings.create(
    model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),     
    input=["warfarin"]
)

chat= client.chat.completions.create(
    model=os.getenv("OPENAI_MODEL", "gpt-4o"),
    messages=[
        {
            "role" : "user",
            "content" : "In one sentence, what is contraindication?"
        },
    ],
)

print(chat.choices[0].message.content)