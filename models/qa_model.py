from transformers import pipeline

def load_model(model_name, device):
    """Load the pre-trained model for text generation."""
    model_pipeline = pipeline("text2text-generation", model=model_name, device=0 if device == "cuda" else -1)
    return model_pipeline

def generate_answer(context, question, model_pipeline, max_new_tokens=50):
    """Generate an answer to the question based on the provided context."""
    # Construct the input prompt
    prompt = (
        f"Use the following context to answer the question:\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    
    # Generate the answer
    output = model_pipeline(prompt, max_new_tokens=max_new_tokens, num_return_sequences=1)
    answer = output[0]["generated_text"]
    
    # Post-process to return only the answer
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    
    return answer