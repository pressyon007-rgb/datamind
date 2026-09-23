import os

from dotenv import load_dotenv
from google import genai


load_dotenv()


API_KEY = os.getenv(
    "GEMINI_API_KEY"
)


def ask_data_analyst(
    question,
    calculated_answer,
    evidence,
    data_context,
    insights=None,
    recommendations=None
):

    if not API_KEY:

        return (
            "Gemini API key is not configured. "
            "Please add GEMINI_API_KEY to the .env file."
        )

    try:

        client = genai.Client(
            api_key=API_KEY
        )

        prompt = f"""

You are an experienced Business Intelligence Analyst.

Answer the user's question using ONLY the uploaded dataset
and the calculated evidence provided below.

Do not invent numbers.

DATA CONTEXT:
{data_context}

CALCULATED ANSWER:
{calculated_answer}

EVIDENCE:
{evidence}

BUSINESS INSIGHTS:
{insights}

RECOMMENDATIONS:
{recommendations}

USER QUESTION:
{question}

Provide:

1. Direct answer
2. Evidence from the data
3. Business meaning
4. Recommended action

Keep the answer practical and understandable to a business manager.

"""

        response = client.models.generate_content(

            model="gemini-3.6-flash",

            contents=prompt
        )

        return response.text

    except Exception as e:

        return f"AI analysis error: {str(e)}"