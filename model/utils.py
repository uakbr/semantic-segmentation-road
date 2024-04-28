import os
import torch

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, filename='checkpoint.pth.tar'):
    """
    Saves the model checkpoint.

    Args:
        model (torch.nn.Module): Trained model.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        epoch (int): Current epoch.
        loss (float): Current training loss.
        checkpoint_dir (str): Directory to save the checkpoint.
        filename (str): Filename of the checkpoint. Default: 'checkpoint.pth.tar'.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(checkpoint, os.path.join(checkpoint_dir, filename))

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    Loads the model checkpoint.

    Args:
        model (torch.nn.Module): Model to load the checkpoint.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): Device to load the checkpoint onto.

    Returns:
        tuple: A tuple containing the loaded model, optimizer, epoch, and loss.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def set_requires_grad(model, requires_grad=False):
    """
    Sets the 'requires_grad' attribute of all parameters in the model.

    Args:
        model (torch.nn.Module): Model to set the 'requires_grad' attribute.
        requires_grad (bool): Whether the parameters require gradient computation. Default: False.
    """
    for param in model.parameters():
        param.requires_grad = requires_grad

def initialize_weights(model):
    """
    Initializes the weights of the model.

    Args:
        model (torch.nn.Module): Model to initialize the weights.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(module.weight, 1)
            torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, 0, 0.01)
            torch.nn.init.constant_(module.bias, 0)