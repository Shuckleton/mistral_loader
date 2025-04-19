from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_cpp import Llama

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
llm = Llama(model_path="models/mistral-7b-v0.1.Q4_K_M.gguf")

# System prompt (defines AI behavior)
system_prompt = "<|start|>system\nYou are a helpful AI assistant.<|end|>\n"

# Chat history storage (limit to last few turns)
chat_history = []


@app.route("/chat", methods=["POST"])
def chat():
    global chat_history

    user_input = request.json.get("message", "")

    # Format the new user input
    user_message = f"<|start|>user\n{user_input}<|end|>\n<|start|>assistant\n"

    # Maintain a chat history window (last 3 interactions)
    chat_history.append(user_message)
    chat_history = chat_history[-3:]  # Keep only the last 3 exchanges

    # Combine system prompt with trimmed chat history
    prompt = system_prompt + "".join(chat_history)

    # Generate AI response
    output = llm(prompt, max_tokens=300, temperature=0.7, top_p=0.9)

    # Extract response
    response = output["choices"][0]["text"].strip()

    # Ensure proper closing to prevent overflow
    response = response.split("<|end|>")[0].strip()

    # Save response in history
    chat_history.append(response + "<|end|>\n")

    # Log user input and AI response
    print(f"User: {user_input}")
    print(f"AI: {response}\n{'-'*50}")

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)
