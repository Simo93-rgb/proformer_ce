import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

class ProformerClassifier(nn.Module):
    def __init__(self, encoder, input_dim, num_classes):
        super(ProformerClassifier, self).__init__()
        self.encoder = encoder
        self.classification_head = ClassificationHead(input_dim, num_classes)

    def forward(self, x, attn_mask):
        encoder_output = self.encoder(x, attn_mask)
        print(f"Encoder output shape: {encoder_output.shape}")  # Debug
        cls_token_output = encoder_output.mean(dim=0)
        print(f"CLS token output shape: {cls_token_output.shape}")  # Debug
        logits = self.classification_head(cls_token_output)
        return logits