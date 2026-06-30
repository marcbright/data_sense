# AI Data Analyst Agent

An AI agent that lets you chat with your data in plain English. Upload a CSV, Excel, or PDF file and ask questions — the agent writes SQL, runs analysis, generates charts, and explains findings automatically.

## Live Demo
(https://data-sense-ai.streamlit.app/)

## Features
- Natural language to SQL/Python query generation
- Self-healing agent (retries failed queries automatically)
- Auto-selected, reasoned chart visualisations
- Proactive insight detection on file upload
- Multi-turn conversation memory
- Downloadable PDF analysis reports
- Full audit trail ("Show my work" on every answer)

## Architecture
Built as a ReAct agent: intent classification → planning → tool execution → evaluation → response composition, with a registry of 8 specialised tools (SQL executor, Python executor, chart generator, insight narrator, anomaly detector, memory retriever, schema inspector, report builder).

## Tech Stack
- LLM: Groq (Llama 3.3 70B)
- Data engine: Pandas, DuckDB
- Visualisation: Plotly
- Frontend: Streamlit
- Memory: ChromaDB
- PDF generation: ReportLab

## Local Setup
1. Clone this repo
2. python -m venv venv && source venv/bin/activate
3. pip install -r requirements.txt
4. cp .env.example .env
5. Add your Groq API key (free at console.groq.com) to .env
6. streamlit run app/main.py

## Deployment
Deployed on Streamlit Community Cloud. Secrets are configured via the Streamlit Cloud dashboard, not .env.

## Author
Bright Osei Kesse — Data Scientist & AI Engineer
LinkedIn: https://www.linkedin.com/in/bright-osei-kesse/
portfolio link: https://bright-osei-kesse.vercel.app/
