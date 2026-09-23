"""
ai_analyst.py - Gemini AI Integration for Custom Data Queries
"""

import os
import google.generativeai as genai


def ask_data_analyst(question, data_context, insights=None, recommendations=None):
    """
    Sends the user's analytical question and dataset context to Gemini.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        return "⚠️ **GEMINI_API_KEY environment variable is missing.** Please set your Gemini API key in your environment or Streamlit secrets to enable AI chat."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        # Construct prompt payload
        prompt = f"""
You are an expert Data Analyst AI collaborator. Analyze the provided dataset overview and answer the user's business question with clear, actionable insights.

### DATASET CONTEXT:
{data_context}

### DATA QUALITY SUMMARY:
{insights if insights else 'No quality metrics supplied.'}

### SYSTEM RECOMMENDATIONS:
{recommendations if recommendations else 'No automated recommendations generated.'}

---

### USER QUESTION:
{question}

---

### INSTRUCTIONS:
- Answer directly based strictly on the provided dataset context and metrics.
- Keep the tone professional, concise, and structured.
- Use bold text for key metrics or findings.
- Offer 2-3 practical next steps or actionable recommendations where appropriate.
"""

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"An error occurred while calling the Gemini API: {str(e)}"
