from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
from pydantic import BaseModel
from dotenv import load_dotenv
from markdown import markdown
from bs4 import BeautifulSoup
import json

# Load environment variables
load_dotenv()

# Initialize FastAPI App
app = FastAPI()

# ✅ Fix CORS: Explicitly allow requests from React frontend (http://localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # ✅ Local React App
        "https://fastapi-ai-planner.onrender.com"  # ✅ Render Backend
    ],
    allow_credentials=True,
    allow_methods=["*"],  # ✅ Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # ✅ Allow all headers
    expose_headers=["*"]  # ✅ Expose headers for preflight requests
)

# Define request model
class ProjectRequest(BaseModel):
    project_name: str
    model: str = "mistral-medium"
    additional_requirements: str = ""

# Health Check Route
@app.get("/")
def home():
    return {"message": "FastAPI is running successfully!"}

# AI Project Plan Generator Route
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
        "Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": request.model,
        "messages": [{"role": "system", "content": prompt}]
    }

    try:
        response = requests.post("https://api.mistral.ai/v1/chat/completions", json=data, headers=headers)
        response.raise_for_status()

        # Extract AI-generated response
        ai_response = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response from AI")

        # Convert Markdown to clean text
        html_response = markdown(ai_response)
        plain_text_response = BeautifulSoup(html_response, "html.parser").get_text()

        try:
            structured_response = json.loads(plain_text_response)
        except json.JSONDecodeError:
            structured_response = {"error": "AI response could not be parsed into JSON."}

        return {
            "project_name": request.project_name,
            "ai_generated_plan": structured_response
        }

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail="Error fetching AI response")
