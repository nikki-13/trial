import torch
from transformers import Swinv2ForImageClassification

# Define model repo and file
MODEL_REPO = "nikki-13/Swin_v2tiny"
MODEL_FILE = "swin_pneumonia_model.pth"

# Download model from Hugging Face Hub
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)

# Initialize model
model = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")

# Load state_dict
state_dict = torch.load(model_path, map_location="cpu")

# 🔍 Debug: Check if the state dict has extra/missing keys
model_keys = set(model.state_dict().keys())
loaded_keys = set(state_dict.keys())

missing_keys = model_keys - loaded_keys
extra_keys = loaded_keys - model_keys

print(f"Missing keys: {missing_keys}")
print(f"Extra keys: {extra_keys}")

# Attempt to load the state dict with strict=False (ignores extra keys)
model.load_state_dict(state_dict, strict=False)

# Set to eval mode
model.eval()

print("✅ Model Loaded Successfully!")
