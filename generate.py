from models.qa_model import load_model, generate_answer
from config import CONFIG
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
    # Load model and tokenizer
model_pipeline = load_model(CONFIG["model_name"], CONFIG["device"])

@app.route("/")
def home():
    return render_template("index.html")

def load_dynamic_context():
    # Load context from a text file or database
    with open("data/context_data.txt", "r") as file:
        return file.read()

@app.route("/chat", methods=["POST"])
def chat():
    context = load_dynamic_context()

    question = request.json.get("message")
    # Generate the answer
    answer = generate_answer(context, question, model_pipeline, CONFIG["max_new_tokens"])
    return jsonify({"response": answer})

if __name__ == "__main__":
    print("\033[92m" + "Server is running successfully on http://127.0.0.1:5000" + "\033[0m")
    app.run(debug=True, port=5000)