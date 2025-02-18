from fastapi import FastAPI, HTTPException
import requests
import logging
import os
from pydantic import BaseModel
from dotenv import load_dotenv
from markdown import markdown
from bs4 import BeautifulSoup

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI App
app = FastAPI()

# Fetch API Key from Environment Variables
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# Logging setup
logging.basicConfig(level=logging.INFO)

# Validate API key availability
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY is missing. Set it in the .env file.")

# Define request model
class ProjectRequest(BaseModel):
    project_name: str
    model: str = "mistral-medium"  # Allow user to specify model
    additional_requirements: str = ""  # Optional additional project details

# Root Route (Fixes 404 Error)
@app.get("/")
def home():
    return {"message": "FastAPI is running successfully!"}

# Generate Project Plan Endpoint
@app.post("/generate_project_plan")
def generate_project_plan(request: ProjectRequest):
    """Generates a detailed AI-driven project plan."""
    
    prompt = f"""
    You are an expert AI assistant that provides structured and properly formatted project plans.
    The project name is: {request.project_name}

    Provide the following details in a structured format:

    1. **Project Overview**  
    2. **Required Components** (Hardware & Software)  
    3. **Step-by-step Implementation Guide**  
    4. **Timeline Estimation**  
    5. **Additional Learning Resources**  

    Make sure the response is well-structured and formatted in plain text.
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
        response.raise_for_status()  # Raises an error for non-2xx responses

        # Extract AI-generated response
        ai_response = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response from AI")

        # Convert Markdown to clean HTML
        html_response = markdown(ai_response)

        # Convert HTML to plain text with formatting
        plain_text_response = BeautifulSoup(html_response, "html.parser").get_text()

        return {
            "project_name": request.project_name,
            "ai_generated_plan": plain_text_response
        }
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Error connecting to Mistral API: {e}")
        raise HTTPException(status_code=500, detail="Error fetching AI response")

    except KeyError:
        logging.error("Unexpected response format from Mistral API")
        raise HTTPException(status_code=500, detail="Unexpected API response format")
