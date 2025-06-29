# src/model.py

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class RGenModel(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(RGenModel, self).__init__()

        # Load CLIP tokenizer and text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

        # Simple generator (you'll replace this with U-Net or Diffusion later)
        self.generator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * 64 * 64),  # Generate 64x64 RGB image
            nn.Tanh()
        )
        self.device = device
        self.to(device)

    def forward(self, prompts):
        tokens = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            text_features = self.text_encoder(**tokens).last_hidden_state[:, 0, :]  # CLS token

        generated = self.generator(text_features)
        return generated.view(-1, 3, 64, 64)  # reshape to image
