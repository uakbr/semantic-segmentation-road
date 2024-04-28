import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101

# Define paths
model_path = 'path/to/trained/model/checkpoint.pth'
video_path = 'path/to/input/video.mp4'
output_path = 'path/to/save/output/video.mp4'

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load trained model
model = deeplabv3_resnet101(num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Load video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Process video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame = transform(frame).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(frame)['out']
    output = output.argmax(1).squeeze().cpu().numpy()
    
    # Postprocess output
    output = Image.fromarray(output.astype(np.uint8), mode='P')
    output.putpalette(palette)
    output = output.convert('RGB')
    output = np.array(output)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    
    # Write output frame
    out.write(output)

cap.release()
out.release()
print(f'Inference complete. Output video saved to: {output_path}')