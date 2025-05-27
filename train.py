import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from model import DeepFakeDetector
from dataset import DeepFakeDataset

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    # Create progress bar for batches
    pbar = tqdm(train_loader, 
                desc=f'Epoch [{epoch}/{num_epochs}] Training',
                leave=False)  
    
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):  # Updated autocast
        for frames, labels in pbar:
            frames = frames.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(frames)
            loss = criterion(outputs.squeeze(), labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['loss'] = total_loss / len(train_loader)
    metrics['labels'] = all_labels
    metrics['preds'] = all_preds
    return metrics

def evaluate(model, test_loader, criterion, device, epoch, num_epochs):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Create progress bar for validation
    pbar = tqdm(test_loader, 
                desc=f'Epoch [{epoch}/{num_epochs}] Validating',
                leave=False)
    
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        for frames, labels in pbar:
            frames = frames.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)
            
            outputs = model(frames)
            loss = criterion(outputs.squeeze(), labels)
            
            total_loss += loss.item()
            probs = torch.sigmoid(outputs.squeeze()).cpu().numpy()
            preds = (probs > 0.5).astype(float)
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['loss'] = total_loss / len(test_loader)
    metrics['probs'] = all_probs
    metrics['labels'] = all_labels
    metrics['preds'] = all_preds
    return metrics

def calculate_metrics(labels, preds):
    """Calculate classification metrics."""
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'f1': f1_score(labels, preds),
        'auc': roc_auc_score(labels, preds)
    }

def plot_confusion_matrix(labels, preds, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(labels, probs, save_path):
    """Plot and save ROC curve."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig(save_path)
    plt.close()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(27)
    np.random.seed(27)
    
    # Speed-optimized hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("WARNING: GPU (CUDA) is not available. Training on CPU will be slower.")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Enable cuDNN auto-tuner
        torch.backends.cudnn.benchmark = True
    
    # Minimal data transforms
    transform = transforms.Compose([
        transforms.Resize((112, 112)),  # Smaller size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Modify dataset to use fewer frames
    train_dataset = DeepFakeDataset(
        root_dir="C:\\Users\\User\\Work\\College\\AIMS\\deepfake\\deepfake-detection\\dataset",
        split='train',
        transform=transform,
        num_frames=8  # Reduced number of frames
    )
    
    # Use a portion of training data as validation set
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Fast training setup
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Simple training setup
    model = DeepFakeDetector().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # Training loop
    best_auc = 0
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(NUM_EPOCHS), desc='Training Progress')
    
    for epoch in epoch_pbar:
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch+1, NUM_EPOCHS)
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, criterion, device, epoch+1, NUM_EPOCHS)
        
        # Update epoch progress bar with metrics
        epoch_pbar.set_postfix({
            'train_loss': f"{train_metrics['loss']:.4f}",
            'train_acc': f"{train_metrics['accuracy']:.4f}",
            'val_loss': f"{test_metrics['loss']:.4f}",
            'val_acc': f"{test_metrics['accuracy']:.4f}"
        })
        
        # Save best model
        if test_metrics['auc'] > best_auc:
            best_auc = test_metrics['auc']
            torch.save(model.state_dict(), 'outputs/best_model.pth')
        
        # Plot metrics
        if epoch == NUM_EPOCHS - 1:
            plot_confusion_matrix(
                test_metrics['labels'],
                test_metrics['preds'],
                'outputs/confusion_matrix.png'
            )
            plot_roc_curve(
                test_metrics['labels'],
                test_metrics['probs'],
                'outputs/roc_curve.png'
            )

if __name__ == '__main__':
    main() 
