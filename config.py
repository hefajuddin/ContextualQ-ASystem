CONFIG = {
    "model_name": "facebook/bart-large-cnn",  # Pre-trained LLM
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # Use GPU if available
    "max_new_tokens": 50,  # Limit length of generated answers
}