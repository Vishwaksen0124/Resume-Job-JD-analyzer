import os
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL_NAME = "mistral-medium"

def build_comparison_prompt(resume: str, jd: str) -> str:
    return f"""
You are a helpful HR assistant. Given the following RESUME and JOB DESCRIPTION, analyze them and respond with:
1. Match Percentage (0-100%)
2. Summary of how well the resume matches the job
3. Key missing skills or qualifications
4. Suggestions to improve the resume for this job

RESUME:
{resume}

JOB DESCRIPTION:
{jd}
"""

def run_analysis(prompt: str) -> str:
    client = Mistral(api_key=MISTRAL_API_KEY)
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.complete(
        model=MISTRAL_MODEL_NAME,
        messages=messages
    )
    return response.choices[0].message.content
