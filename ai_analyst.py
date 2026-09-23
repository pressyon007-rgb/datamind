import os
import streamlit as st

# Safe import for python-dotenv (prevents crashes on Streamlit Cloud)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Safe import for google-genai
try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


def _get_api_key():
    """Fetch API key from Streamlit Secrets or Environment Variables."""
    # 1. Try Streamlit Cloud secrets first
    if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    # 2. Fall back to local .env environment variable
    return os.getenv("GEMINI_API_KEY")


def ask_data_analyst(
    question,
    calculated_answer=None,
    evidence=None,
    data_context=None,
    insights=None,
    recommendations=None
):
    if not HAS_GENAI:
        return (
            "⚠️ Google GenAI SDK (`google-genai`) is not installed. "
            "Please add `google-genai>=0.1.0` to `requirements.txt`."
        )

    api_key = _get_api_key()

    if not api_key:
        return (
            "⚠️ Gemini API key is not configured. "
            "Please set `GEMINI_API_KEY` in Streamlit Cloud Secrets or your `.env` file."
        )

    try:
        client = genai.Client(api_key=api_key)

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
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:
        return f"AI analysis error: {str(e)}"
