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

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='Training')
    for frames, labels in pbar:
        frames = frames.to(device)
        labels = labels.float().to(device)
        
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs.squeeze(), labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = (outputs.squeeze() > 0.5).float()
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['loss'] = total_loss / len(train_loader)
    return metrics

def evaluate(model, test_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for frames, labels in tqdm(test_loader, desc='Evaluating'):
            frames = frames.to(device)
            labels = labels.float().to(device)
            
            outputs = model(frames)
            loss = criterion(outputs.squeeze(), labels)
            
            total_loss += loss.item()
            probs = outputs.squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(float)
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['loss'] = total_loss / len(test_loader)
    metrics['probs'] = all_probs
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
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Hyperparameters
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    USE_SR = False  # Set to True to enable super-resolution
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = DeepFakeDataset(
        root_dir='dataset',
        split='train',
        use_sr=USE_SR,
        transform=transform
    )
    
    test_dataset = DeepFakeDataset(
        root_dir='dataset',
        split='test',
        use_sr=USE_SR,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    model = DeepFakeDetector(use_sr=USE_SR).to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # Training loop
    best_auc = 0
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        print('Training metrics:', train_metrics)
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, criterion, device)
        print('Test metrics:', test_metrics)
        
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