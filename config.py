CONFIG = {
    "model_name": "facebook/bart-large-cnn",  # Pre-trained LLM
    # "model_name": "okanvk/bert-question-answering-cased-squadv2_tr",  # Pre-trained LLM
    
    # "device": "cuda" if torch.cuda.is_available() else "cpu",  # Use GPU if available
    "device": "cuda",
    "max_new_tokens": 100,  # Limit length of generated answers
}