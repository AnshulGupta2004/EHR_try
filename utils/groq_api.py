import os
import time  # Import time module for sleep function
from groq import Groq  # Replace with the actual import if different

class GroqAPI:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = "mixtral-8x7b-32768"  # Update if necessary
        self.system_prompt = {
            "role": "system",
            "content": "You are a helpful assistant specialized in generating SQL queries from natural language questions."
        }

    def generate_sql(self, prompt, max_tokens=500, temperature=1.2):
        chat_history = [
            self.system_prompt,
            {"role": "user", "content": prompt}
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=chat_history,
            max_tokens=max_tokens,
            temperature=temperature
        )

        result = response.choices[0].message.content.strip()

        # Introduce a 1-second delay before the function can be called again
        time.sleep(1)

        return result
