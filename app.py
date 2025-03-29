from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import requests  # Add this import

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class LLMInterface:
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.system_message = """You are an AI that generates high-quality and visually appealing frontend code in React.js using MUI. Follow these strict guidelines:
        - **Design Aesthetic**: Ensure a modern, and beautiful design.
        - **Typography**: Use only the 'Fira Sans' font.
        - **Color Palette**: Apply these colors strictly:
          - Primary: #1B374C
          - Accent: #F39325
          - Background: #F5F5F6
          - Text: #000000 or #FFFFFF
        """
        self.model = "Qwen/Qwen2.5-Coder-7B-Instruct-fast"
    
    def query(self, prompt: str) -> str:
        payload = {
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "model": self.model
        }
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code}, {response.text}"

@app.route('/generate', methods=['POST'])
def generate_code():
    data = request.get_json()
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    api_url = "https://router.huggingface.co/nebius/v1/chat/completions"
    api_key = "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"  # Replace with your HuggingFace API key
    llm = LLMInterface(api_url, api_key)
    response = llm.query(prompt)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
