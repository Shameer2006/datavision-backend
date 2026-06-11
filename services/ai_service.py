"""
AI Service
Handles communication with the Google Gemini 2.5 Flash model.
Generates text overviews and Plotly chart configurations from dataset schemas.
"""

import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")


SYSTEM_PROMPT = """You are DataVision AI, an expert data analyst assistant.
You help users understand, analyze, and visualize their data.

When the user uploads a dataset, you will receive the dataset's schema including:
- Column names and their data types
- Number of rows
- A sample of the data rows

Your job is to answer the user's questions about their data.

WHEN TO INCLUDE A CHART vs TEXT ONLY:
- If the user asks to "visualize", "plot", "chart", "graph", "show me a chart", or explicitly requests a visualization → include "plotly_config" with a valid Plotly chart.
- If the user asks informational questions like "what columns are there?", "what does the data contain?", "how many rows?", "describe the data", "what values are in column X?" → respond with ONLY "text_overview" and set "plotly_config" to null.
- If the user uploads a file without a specific question, provide a brief text summary of the dataset, but do NOT auto-generate a chart.

TEXT FORMATTING RULES (very important):
- Use PLAIN TEXT only. Do NOT use markdown formatting like **bold**, *italic*, ##headers, or any other markdown syntax.
- Use bullet points (•) for listing items.
- Use simple labels followed by a colon for key-value pairs (e.g., "Total Rows: 9994").
- Keep responses concise and clean. No need for excessive detail.
- Use line breaks (\\n) to separate sections.
- Example format for a dataset summary:

"This dataset contains sales data with 9994 rows and 25 columns.\\n\\nKey columns:\\n• Order ID - Unique order identifier\\n• Order Date - Date of order\\n• Sales - Sale amount\\n• Profit - Profit amount\\n\\nYou can ask me to visualize trends, compare categories, or explore any specific column."

YOU MUST ALWAYS respond with a valid JSON object (no markdown, no code fences, just raw JSON):

For text-only responses:
{
  "text_overview": "Plain text response here with bullet points using •",
  "plotly_config": null
}

For visualization responses:
{
  "text_overview": "Brief explanation of the chart...",
  "plotly_config": {
    "data": [
      {
        "type": "bar",
        "x": ["A", "B", "C"],
        "y": [10, 20, 30],
        "marker": { "color": ["#6366f1", "#8b5cf6", "#a78bfa"] }
      }
    ],
    "layout": {
      "title": { "text": "Chart Title" },
      "template": "plotly_white",
      "paper_bgcolor": "rgba(0,0,0,0)",
      "plot_bgcolor": "rgba(0,0,0,0)",
      "font": { "family": "Inter, sans-serif" },
      "margin": { "l": 50, "r": 30, "t": 50, "b": 50 }
    }
  }
}

IMPORTANT RULES:
- NEVER use markdown formatting (no **, no *, no ##, no backticks).
- Only include "plotly_config" when the user explicitly requests a chart or visualization.
- For informational queries, set "plotly_config" to null and give a detailed plain text response with bullet points (•).
- When generating charts, use modern, vibrant color palettes (purples, indigos, teals, etc.).
- Use "plotly_white" template and transparent backgrounds for charts.
- DO NOT wrap the JSON in markdown code fences. Return ONLY the raw JSON object.
"""


async def generate_chat_response(
    user_message: str,
    schema: dict | None = None,
    df_json: str | None = None,
    model_name: str = "DataVision Flash"
) -> dict:
    """
    Send a prompt to Gemini and return the parsed response.

    Args:
        user_message: The user's chat message.
        schema: Optional dict with column names, dtypes, row_count, sample_data.
        df_json: Optional full dataset as JSON string (for computation).
        model_name: The selected model name (DataVision Flash or DataVision Pro).

    Returns:
        dict with 'text_overview' and optionally 'plotly_config'.
    """
    # Build the context prompt
    parts = [SYSTEM_PROMPT]

    if schema:
        context = (
            f"\n\n--- DATASET INFO ---\n"
            f"Filename: {schema['filename']}\n"
            f"Rows: {schema['row_count']}\n"
            f"Columns ({len(schema['columns'])}): {', '.join(schema['columns'])}\n"
            f"Data Types: {json.dumps(schema['dtypes'], indent=2)}\n"
            f"Sample Data (first 5 rows): {json.dumps(schema['sample_data'], indent=2, default=str)}\n"
            f"--- END DATASET INFO ---\n"
        )
        parts.append(context)

        if df_json:
            parts.append(f"\n--- DATASET SAMPLE (JSON, first 50 rows) ---\n{df_json}\n--- END DATASET SAMPLE ---\n")

    parts.append(f"\nUser Message: {user_message}")

    prompt = "\n".join(parts)

    try:
        # Determine model id dynamically
        gemini_model_id = "gemini-3-flash" if model_name == "DataVision Pro" else "gemini-2.5-flash"
        generative_model = genai.GenerativeModel(gemini_model_id)

        response = await generative_model.generate_content_async(prompt)
        raw_text = response.text.strip()

        # Strip markdown code fences if the model wraps them anyway
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1] if "\n" in raw_text else raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3].strip()
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].strip()

        result = json.loads(raw_text)
        return result

    except json.JSONDecodeError:
        # If JSON parsing fails, return just the text
        return {
            "text_overview": response.text if 'response' in dir() else "I encountered an error processing your request.",
            "plotly_config": None,
        }
    except Exception as e:
        return {
            "text_overview": f"Error communicating with AI: {str(e)}",
            "plotly_config": None,
        }
