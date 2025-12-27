from openai import OpenAI
import os

# Create OpenAI client (reads OPENAI_API_KEY from env)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_answer(question: str, context: list[str]) -> str:
    """
    Takes:
      - user question
      - retrieved context chunks
    Returns:
      - final natural language answer
    """

    # Join retrieved chunks into a single context block
    context_text = "\n\n".join(context)

    prompt = f"""
You are a professional assistant answering questions about Selim Erenay
using ONLY the information provided below.

Context:
{context_text}

Question:
{question}

Answer clearly and concisely.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful portfolio assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content