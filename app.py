import torch
from transformers import Swinv2ForImageClassification

# Define model repo and file
MODEL_REPO = "nikki-13/Swin_v2tiny"
MODEL_FILE = "swin_pneumonia_model.pth"

# Download model from Hugging Face Hub
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)

# Initialize the model architecture
model = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")

# Load the state dictionary
state_dict = torch.load(model_path, map_location="cpu")

# If the state_dict is inside a key (e.g., "model"), extract it
if isinstance(state_dict, dict) and "model" in state_dict:
    state_dict = state_dict["model"]

# Load state_dict into model
model.load_state_dict(state_dict)

# Set model to eval mode
model.eval()

print("✅ Model Loaded Successfully!")
