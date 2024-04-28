import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet101

# Define hyperparameters
batch_size = 16
learning_rate = 0.007
num_epochs = 50
val_split = 0.2

# Define paths
data_dir = 'path/to/preprocessed/data'
checkpoint_dir = 'path/to/save/model/checkpoints'

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load preprocessed data
train_dataset = CustomCityscapesDataset(os.path.join(data_dir, 'train'))
val_dataset = CustomCityscapesDataset(os.path.join(data_dir, 'val'))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define model
model = deeplabv3_resnet101(num_classes=num_classes)
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=num_epochs, power=0.9)

# Train model
best_val_loss = float('inf')
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    
    # Train
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
    train_loss = train_loss / len(train_dataset)
    
    # Validate  
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)['out']
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
    val_loss = val_loss / len(val_dataset)
    
    print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
    
    # Save checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'best_model_epoch{epoch+1}.pth'))
        
    scheduler.step()

print('Training complete!')