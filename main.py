from fastapi import FastAPI, HTTPException
import requests
import logging
import os
import json
from pydantic import BaseModel
from dotenv import load_dotenv
from markdown import markdown
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# Initialize FastAPI App
app = FastAPI()

# Fetch API Key
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# Define request model
class ProjectRequest(BaseModel):
    project_name: str
    model: str = "mistral-medium"
    additional_requirements: str = ""

@app.get("/")
def home():
    return {"message": "FastAPI is running successfully!"}

@app.post("/generate_project_plan")
def generate_project_plan(request: ProjectRequest):
    """Generates a structured AI-driven project plan."""
    
    prompt = f"""
    You are an AI expert generating structured project plans.
    Project: {request.project_name}

    Structure:
    1. Project Overview
    2. Required Components (Hardware & Software)
    3. Implementation Steps
    4. Timeline Estimate
    5. Additional Learning Resources

    Provide responses as a structured JSON object.
    """

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": request.model,
        "messages": [{"role": "system", "content": prompt}]
    }

    try:
        response = requests.post(MISTRAL_API_URL, json=data, headers=headers)
        response.raise_for_status()

        # Extract AI-generated response
        ai_response = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response from AI")

        # Convert Markdown to clean text
        html_response = markdown(ai_response)
        plain_text_response = BeautifulSoup(html_response, "html.parser").get_text()

        # Ensure AI response is structured properly
        try:
            structured_response = json.loads(plain_text_response)  # Convert AI output to JSON
        except json.JSONDecodeError:
            structured_response = {"error": "AI response could not be parsed into JSON."}

        return {
            "project_name": request.project_name,
            "ai_generated_plan": structured_response
        }

    except requests.exceptions.RequestException as e:
        logging.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail="Error fetching AI response")
