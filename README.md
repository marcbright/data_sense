# AI Data Analyst Agent

An AI-powered data analyst that lets you upload CSV, Excel, or PDF files and ask questions about your data in plain English. The agent generates SQL/Python code, executes analysis, creates visualizations, and explains findings.

## Phase 1: Foundation
This repository contains the core configuration, utility modules, and folder structure for the AI Data Analyst Agent.

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-data-analyst
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   - Copy `.env.example` to `.env`
   - Add your Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Customize other settings like `MODEL_NAME` or `MAX_FILE_SIZE_MB` if needed.
      - Alternatively, set `LLM_PROVIDER=groq` and provide the following variables to use a Groq-compatible endpoint:
         - `GROQ_API_KEY` — your Groq API key
         - `GROQ_API_URL` — the full HTTP(s) inference endpoint for your Groq provider
         - When using `LLM_PROVIDER=groq`, ensure `GROQ_API_URL` accepts a JSON payload with `prompt` and returns a textual `text`/`output` field (see `config.py` for the minimal expected shape).

5. **Run the application**
   ```bash
   streamlit run app/main.py
   ```

## Tech Stack
- **AI Engine**: Google Gemini (via `google-generativeai`)
- **Web Interface**: Streamlit
- **Data Engine**: DuckDB & Pandas
- **Visualization**: Plotly
- **Document Parsing**: PyMuPDF
- **Vector Memory**: ChromaDB
- **Validation**: Pydantic
