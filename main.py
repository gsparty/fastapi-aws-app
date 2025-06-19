from fastapi import FastAPI, Form, Request
from pydantic import BaseModel
from mangum import Mangum
from dotenv import load_dotenv
import os
import requests
from fastapi.staticfiles import StaticFiles

# Load environment variables
load_dotenv()
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Initialize FastA                                                                                                                                          PI app and handler
app = FastAPI()
handler = Mangum(app)

# Define request models
class PromptRequest(BaseModel):
    topic: str
    level: str
    style: str

class FollowupInput(BaseModel):
    topic: str
    level: str
    question: str

# Define endpoints
@app.post("/explain")
def explain(req: PromptRequest):
    messages = []
    if req.style == "few-shot":
        messages = [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": "What is gravity?"},
            {"role": "assistant", "content": "Gravity is the force that pulls things together."},
            {"role": "user", "content": f"Explain '{req.topic}' to a {req.level} student."}
        ]
    else:
        messages = [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": generate_prompt(req.topic, req.level, req.style)}
        ]

    payload = {
        "model": "sonar",
        "messages": messages,
        "stream": False
    }

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    url = "https://api.perplexity.ai/chat/completions"

    try:
        response = requests.post(url, headers=headers, json=payload)
        data = response.json()

        if "choices" not in data:
            return {"error": f"Unexpected response: {data}"}

        return {
            "prompt_used": messages,
            "response": data['choices'][0]['message']['content']
        }

    except Exception as e:
        return {"error": str(e)}

def generate_prompt(topic, level, style):
    levels = {
        "5yo": "Explain like a bedtime story with simple language.",
        "highschool": "Use relatable analogies and basic science.",
        "university": "Use technical academic language."
    }

    if style == "basic":
        return f"Explain the topic '{topic}' to a {level} student. {levels.get(level, '')}"
    elif style == "instruction":
        return f"You are a teacher. Explain '{topic}' to a {level} student. {levels.get(level, '')}"
    elif style == "few-shot":
        return f"""Q: What is gravity?\nA: Gravity is the force that pulls things together.\n\nQ: Explain '{topic}' to a {level} student.\nA:"""
    else:
        return f"Explain '{topic}' simply."

@app.post("/followup")
def followup(req: PromptRequest):
    prompt = f"Based on the topic '{req.topic}', ask a thoughtful follow-up question appropriate for a {req.level} student."
    return simple_response(prompt)

@app.post("/followup_answer")
def followup_answer(req: FollowupInput):
    prompt = f"The user studied '{req.topic}' at {req.level} level. They asked: '{req.question}'. Please answer clearly and concisely."
    return simple_response(prompt)

@app.post("/summarize")
def summarize(req: PromptRequest):
    prompt = f"Summarize the topic '{req.topic}' in a concise way for a {req.level} student."
    return simple_response(prompt)

@app.post("/counter")
def counter(req: PromptRequest):
    prompt = f"Give a counterargument or alternate perspective on the topic '{req.topic}', suitable for a {req.level} student."
    return simple_response(prompt)

@app.post("/adjust_level")
def adjust_level(req: PromptRequest, direction: str = "simpler"):
    levels = ["5yo", "highschool", "university"]
    try:
        current_index = levels.index(req.level)
    except ValueError:
        return {"error": f"Invalid level: {req.level}"}

    if direction == "simpler" and current_index > 0:
        new_level = levels[current_index - 1]
    elif direction == "harder" and current_index < len(levels) - 1:
        new_level = levels[current_index + 1]
    else:
        new_level = req.level  # Can't go further

    # Generate updated explanation
    prompt = generate_prompt(req.topic, new_level, req.style)
    return simple_response(prompt)

def simple_response(prompt: str):
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    url = "https://api.perplexity.ai/chat/completions"

    try:
        response = requests.post(url, headers=headers, json=payload)
        data = response.json()

        if "choices" not in data:
            return {"error": f"Unexpected response: {data}"}

        return {
            "prompt_used": prompt,
            "response": data['choices'][0]['message']['content']
        }

    except Exception as e:
        return {"error": str(e)}

# Mount static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")