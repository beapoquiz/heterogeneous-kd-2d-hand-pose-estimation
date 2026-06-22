import torch, sys, os
sys.path.append(os.path.join(os.getcwd(), 'student_model'))
from blazehand_landmark import BlazeHandLandmark

model = BlazeHandLandmark()
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total parameters:     {total:,}')
print(f'Trainable parameters: {trainable:,}')
print(f'Model size:           {total*4/1024/1024:.2f} MB (float32)')