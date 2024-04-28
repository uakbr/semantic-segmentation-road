import os
import numpy as np
from PIL import Image
from torchvision import transforms

# Define paths
data_dir = 'path/to/cityscapes/dataset'
output_dir = 'path/to/save/preprocessed/data'

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Loop through dataset
for split in ['train', 'val']:
    # Create directories
    os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # Loop through images
    for img_file in os.listdir(os.path.join(data_dir, split, 'images')):
        # Load and preprocess image
        img_path = os.path.join(data_dir, split, 'images', img_file)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        
        # Save preprocessed image
        out_img_path = os.path.join(output_dir, split, 'images', img_file)
        np.save(out_img_path.replace('.png', '.npy'), img_tensor.numpy())
        
    # Loop through labels  
    for label_file in os.listdir(os.path.join(data_dir, split, 'labels')):
        # Load label
        label_path = os.path.join(data_dir, split, 'labels', label_file)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        
        # Convert to one-hot encoding
        label = np.eye(num_classes)[label]
        
        # Save preprocessed label
        out_label_path = os.path.join(output_dir, split, 'labels', label_file)  
        np.save(out_label_path.replace('.png', '.npy'), label)

print(f'Preprocessing complete. Preprocessed data saved to: {output_dir}')