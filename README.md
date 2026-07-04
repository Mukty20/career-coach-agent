# Career Coach Agent 🎯

A single-agent AI application that helps Nigerian tech students and early-career professionals with career guidance, CV/resume feedback, skill-gap analysis, and job search strategy — grounded in the Nigerian job market.

**Live App:** [https://mukty20-career-coach-agent-app-ypdfsm.streamlit.app/](https://mukty20-career-coach-agent-app-ypdfsm.streamlit.app/)
**Demo Video:** [Add your Google Drive video URL here]

---

## The Problem

Nigerian tech students and bootcamp graduates often struggle to get personalized, current career guidance. Generic advice online doesn't account for the local job market (salary ranges, in-demand roles, common entry points), and most students don't have access to a career coach who can review their actual CV and give specific, actionable feedback. This agent fills that gap — combining an LLM with real CV parsing, live web search, and a curated knowledge base of Nigerian career resources.

## How It Works

This is a **single-agent architecture** built on the Groq API. The agent reasons about each user message and decides, on its own, whether to answer directly, search the web, or lean on retrieved knowledge base context — there's no separate router or multi-agent hand-off.

**Components:**

| Component | Purpose |
|---|---|
| **LLM** | Groq API (`openai/gpt-oss-120b`) — the reasoning core, free tier |
| **Tool 1: CV Analyzer** | Extracts text from an uploaded PDF CV (`pypdf`) so the agent can ground advice in the user's actual background |
| **Tool 2: Web Search** | Live search (`ddgs`) for current job listings, salary data, or market trends — keeps advice from going stale |
| **RAG** | A small curated knowledge base (Nigerian CV best practices, tech job market data, career path guidance) retrieved via TF-IDF + cosine similarity (`scikit-learn`) and injected as grounding context |
| **Memory** | Conversation history persisted in Streamlit session state, so the agent remembers what's already been discussed in a session |

**Flow:** User sends a message → relevant knowledge base chunks are retrieved via RAG → if a CV is uploaded, its parsed text is included as context → the model decides whether it needs to call the `web_search` tool or can answer directly → response streamed back to the user.

## Project Structure

```
career-coach-agent/
├── app.py                  # Streamlit UI and session state
├── agent.py                # Core agent loop (Groq tool-use)
├── tools/
│   ├── cv_analyzer.py      # PDF text extraction
│   ├── web_search.py       # DuckDuckGo search wrapper
│   └── rag.py               # TF-IDF retrieval over knowledge base
├── knowledge_base/          # Curated markdown files for RAG
│   ├── cv_resume_tips.md
│   ├── nigerian_tech_market.md
│   └── career_paths_cs.md
├── requirements.txt
└── .streamlit/
    └── secrets.toml.example
```

## Running Locally

1. Clone the repo and install dependencies:
   ```bash
   git clone <your-repo-url>
   cd career-coach-agent
   pip install -r requirements.txt
   ```

2. Add your Groq API key:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   # then edit .streamlit/secrets.toml and paste in your key
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Deploying to Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app pointing at this repo, with `app.py` as the entry point.
3. In the app's **Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your-key-here"
   ```
4. Deploy. That's it.

## Tech Stack

- **LLM:** Groq API (openai/gpt-oss-120b)
- **Frontend/Deployment:** Streamlit
- **PDF Parsing:** pypdf
- **Web Search:** ddgs (DuckDuckGo Search)
- **RAG:** scikit-learn (TF-IDF + cosine similarity)

## Author

Built by Muktar ([@mukty_codes](https://instagram.com/mukty_codes)) as a capstone project for an AI/ML bootcamp.
