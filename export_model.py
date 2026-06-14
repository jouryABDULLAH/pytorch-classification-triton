# converts .pt to TorchScript

import torch
import torchvision.models as models


model = models.resnet50(pretrained=True)
model.eval()

# Trace it
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model_repository/fundus_classifier/1/model.pt")

print("Model exported successfully")