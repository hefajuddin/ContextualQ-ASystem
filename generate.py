import torch
from utils.data_loader import load_context
from models.qa_model import load_model, generate_answer
from config import CONFIG

def main():
    # Load context from JSON
    context_data = load_context("data/context_data.json")
    context = context_data["context"]

    # Define a sample question
    question = "Who is Miton?"

    # Load model and tokenizer
    model_pipeline = load_model(CONFIG["model_name"], CONFIG["device"])

    # Generate the answer
    answer = generate_answer(context, question, model_pipeline, CONFIG["max_new_tokens"])

    # Print the answer
    print(f"Question: {question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()