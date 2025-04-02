from flask import Flask, request, jsonify
from flask_cors import CORS 
import requests 
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app) 

class LLMInterface:
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.system_message = """You are an AI that generates only JSON code for frontend components in React.js using MUI. Follow these strict guidelines:
        - **Format**: Output only JSON, no explanations, no additional text.
        - **Structure**: Each component should be an object with:
          - "type": The MUI component type (e.g., "container", "typography", "box").
          - "props": A dictionary containing the properties of the component.
          - "children": Either a string (for text) or a list of child components.
        - **Design Aesthetic**: Follow these styles:
          - Primary color: #1B374C
          - Accent color: #F39325
          - Background: #F5F5F6
          - Typography: 'Fira Sans' font
        - **Strict Rules**: Only return a valid JSON object. Do not include any additional text.
        """
        self.model = "Qwen/Qwen2.5-Coder-7B-Instruct-fast"
    
    def query(self, prompt: str) -> str:
        payload = {
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "model": self.model
        }
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code}, {response.text}"

# Route de base
@app.route('/')
def home():
    return "Welcome to the AI Code Generator API!"

@app.route('/generate', methods=['POST'])
def generate_code():
    data = request.get_json()
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    api_url = "https://router.huggingface.co/nebius/v1/chat/completions"
    api_key = os.getenv("HF_API_KEY")  
    
    if not api_key:
        return jsonify({"error": "API key is missing. Set it in the .env file."}), 500
    
    llm = LLMInterface(api_url, api_key)
    response = llm.query(prompt)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
