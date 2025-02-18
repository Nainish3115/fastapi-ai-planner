from fastapi import FastAPI, HTTPException
import requests
import logging
import os
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI App
app = FastAPI()

# Fetch Mistral API Key from Environment Variables
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

@app.post("/generate_project_plan")
def generate_project_plan(request: ProjectRequest):
    """Generates a detailed AI-driven project plan."""
    
    # Construct the AI prompt
    prompt = f"""
You are an expert AI assistant that provides structured and properly formatted project plans. 
The project name is: {request.project_name}.

Provide the following details in a **strictly formatted** plain-text format with clear numbered sections:

1. **Project Overview**  
   - Summarize the project in 2-3 simple sentences.

2. **Required Components**  
   - Clearly separate **Hardware** and **Software** components.
   - Use a **plain-text numbered list** (no markdown, no bullet points).

3. **Step-by-step Implementation Guide**  
   - Write each step **on a new line** with **clear numbering** (e.g., "1.", "2.", "3.").
   - Each step must be **simple, direct, and formatted properly**.

4. **Timeline Estimation**  
   - Provide an estimated duration in **weeks/months** based on project complexity.
   - Format this section in plain text with proper paragraph spacing.

5. **Additional Learning Resources**  
   - List useful **URLs in plain text** (no markdown formatting).
   - Separate links **clearly** to avoid merging.

Make sure to follow the formatting rules **strictly** and avoid using markdown-style formatting like bullet points, asterisks, or unnecessary indentation.
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
        response.raise_for_status()  # Raise error for non-2xx responses
        ai_response = response.json()

        # Extract AI-generated content safely
        return {
            "project_name": request.project_name,
            "ai_generated_plan": ai_response.get("choices", [{}])[0].get("message", {}).get("content", "No response from AI")
        }
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Error connecting to Mistral API: {e}")
        raise HTTPException(status_code=500, detail="Error fetching AI response")

    except KeyError:
        logging.error("Unexpected response format from Mistral API")
        raise HTTPException(status_code=500, detail="Unexpected API response format")
