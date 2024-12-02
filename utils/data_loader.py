import json

def load_context(file_path):
    """Load context data from a JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data