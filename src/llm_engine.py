import os
import google.generativeai as genai
import json
from typing import List, Dict

from tenacity import retry, stop_after_attempt, wait_exponential

class LLMEngine:
    def __init__(self):
   
        self.model = genai.GenerativeModel('gemini-2.0-flash-lite-001')

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_response(self, query: str, context_chunks: List[str]) -> str:
        """Generates a response based on the query and retrieved context."""
        
        context_text = "\n\n".join(context_chunks)
        
       
        prompt = f"""
        You are a helpful assistant. Answer the user's question based ONLY on the provided context.
        If the answer is not in the context, say "I don't have enough information to answer that."
        
        Context:
        {context_text}
        
        User Question: {query}
        
        Answer:
        """
        
        response = self.model.generate_content(prompt)
        return response.text

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=5, max=60))
    def extract_memory(self, user_input: str, bot_response: str) -> Dict[str, str]:
        """Analyzes the interaction to extract persistent memories."""
        
        prompt = f"""
        Analyze the following interaction and extract high-signal facts if present.
        
        User: {user_input}
        Bot: {bot_response}
        
        Rules:
        1. "user_memory": Extract permanent user facts (e.g., "User is a Project Manager"). Return "" if none.
        2. "company_memory": Extract reusable org-wide learnings (e.g., "Project Finance uses tool X"). Return "" if none.
        
        Return JSON format:
        {{
            "user_memory": "...",
            "company_memory": "..."
        }}
        """
        
        response = self.model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text)
