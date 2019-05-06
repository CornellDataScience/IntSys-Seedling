import torch
import sys

model_in = sys.argv[1]
model_out = sys.argv[2]

model = torch.load(model_in, map_location='cpu')
model = model.load_state_dict(model)
model.eval()
dummy_input = torch.randn(1, 3, 128, 128)
if torch.cuda.is_available():
    dummy_input = dummy_input.to(device)
torch.onnx.export(model, dummy_input, model_out)