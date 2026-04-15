# FastAPI AWS App

FastAPI backend for explanation-style prompts using the Perplexity API (`sonar` model), designed to run locally or on AWS Lambda via Mangum.

## What it does

- Generate explanations for a topic at different learner levels
- Offer follow-up questions and answers
- Summarize topics
- Generate counterarguments
- Adjust explanation level (`5yo`, `highschool`, `university`)

## API endpoints

- `POST /explain`
- `POST /followup`
- `POST /followup_answer`
- `POST /summarize`
- `POST /counter`
- `POST /adjust_level`

## Project files

- `main.py` / `application.py` - FastAPI app and API routes
- `static/` - static frontend pages
- `requirements.txt` - dependencies
- `Procfile` - deployment entry command

## Local setup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```env
PERPLEXITY_API_KEY=your_key_here
```

Run:

```bash
uvicorn main:app --reload
```
