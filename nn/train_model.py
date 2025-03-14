import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler

from model import ChessCNN

# Use CPU
device = torch.device("cpu")
print(f"Default device: CPU")

# Try GPU
try:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU")
except Exception as e:
    print(f"Error initializing GPU: {e}. Using CPU instead.")

# Enhanced ChessCNN model with additional features
class EnhancedChessCNN(nn.Module):
    """Improved chess model"""
    def __init__(self, input_channels=12, num_filters=128, num_residual_blocks=10, num_output_moves=1968):
        super(EnhancedChessCNN, self).__init__()
        
        # Initial layers
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        
        # Bottleneck blocks
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_residual_blocks):
            block = nn.Sequential(
                # Narrow down
                nn.Conv2d(num_filters, num_filters//2, kernel_size=1),
                nn.BatchNorm2d(num_filters//2),
                nn.ReLU(),
                # Process features
                nn.Conv2d(num_filters//2, num_filters//2, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_filters//2),
                nn.ReLU(),
                # Expand back
                nn.Conv2d(num_filters//2, num_filters, kernel_size=1),
                nn.BatchNorm2d(num_filters),
            )
            self.residual_blocks.append(block)
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Move predictor
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.3),  # Prevent overfitting
            nn.Linear(64 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_output_moves)
        )
        
        # Evaluation head
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Initial conv
        x = self.initial_conv(x)
        
        # Apply blocks
        for block in self.residual_blocks:
            residual = x
            out = block(x)
            x = F.relu(out + residual)  # Skip connection
        
        # Apply attention
        att = self.attention(x)
        x = x * att
        
        # Get outputs
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value

def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    """Handle class imbalance"""
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()

def train_model(model, train_loader, val_loader, optimizer, policy_criterion, value_criterion, 
                num_epochs=40, scheduler=None, early_stopping_patience=12, model_save_path='./models/final'):
    """Train chess model"""
    # Make directory
    os.makedirs(model_save_path, exist_ok=True)
    
    # Init tracking
    best_val_loss = float('inf')
    best_policy_acc = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    policy_accuracy_history = []
    value_mae_history = []
    
    # Start training
    print("\nStarting FINAL OPTIMIZED training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_policy_acc = 0
        train_value_mae = 0
        batch_count = 0
        
        # Show progress
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for positions, values, moves in train_pbar:
            # Move data
            positions = positions.to(device)
            values = values.to(device).float()
            moves = moves.to(device).long()
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            policy_output, value_output = model(positions)
            
            # Fix shapes
            if value_output.shape != values.shape:
                if len(values.shape) == 1:
                    values = values.unsqueeze(1)
                elif len(value_output.shape) == 1:
                    value_output = value_output.unsqueeze(1)
            
            # Calculate loss
            if epoch < 5:
                # Start stable
                policy_loss = policy_criterion(policy_output, moves)
            else:
                # Use focal
                policy_loss = focal_loss(policy_output, moves, gamma=2.0)
            
            value_loss = value_criterion(value_output, values)
            
            # Combined loss
            policy_weight = min(0.9, 0.7 + epoch * 0.01)  # Increase gradually
            loss = policy_weight * policy_loss + (1 - policy_weight) * value_loss
            
            # Backprop
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            
            # Check accuracy
            _, predicted = torch.max(policy_output, 1)
            train_policy_acc += (predicted == moves).sum().item() / moves.size(0)
            
            # Check error
            train_value_mae += torch.abs(value_output - values).mean().item()
            
            batch_count += 1
            
            # Update bar
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'policy_acc': f"{(predicted == moves).sum().item() / moves.size(0):.4f}",
                'value_mae': f"{torch.abs(value_output - values).mean().item():.4f}"
            })
        
        # Get averages
        train_loss /= batch_count
        train_policy_acc /= batch_count
        train_value_mae /= batch_count
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_policy_acc = 0
        val_value_mae = 0
        batch_count = 0
        
        # Show progress
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        
        with torch.no_grad():
            for positions, values, moves in val_pbar:
                # Move data
                positions = positions.to(device)
                values = values.to(device).float()
                moves = moves.to(device).long()
                
                # Forward pass
                policy_output, value_output = model(positions)
                
                # Fix shapes
                if value_output.shape != values.shape:
                    if len(values.shape) == 1:
                        values = values.unsqueeze(1)
                    elif len(value_output.shape) == 1:
                        value_output = value_output.unsqueeze(1)
                
                # Calculate loss
                policy_loss = policy_criterion(policy_output, moves)
                value_loss = value_criterion(value_output, values)
                
                policy_weight = min(0.9, 0.7 + epoch * 0.01)
                loss = policy_weight * policy_loss + (1 - policy_weight) * value_loss
                
                # Track metrics
                val_loss += loss.item()
                
                # Check accuracy
                _, predicted = torch.max(policy_output, 1)
                val_policy_acc += (predicted == moves).sum().item() / moves.size(0)
                
                # Check error
                val_value_mae += torch.abs(value_output - values).mean().item()
                
                batch_count += 1
                
                # Update bar
                val_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}", 
                    'policy_acc': f"{(predicted == moves).sum().item() / moves.size(0):.4f}",
                    'value_mae': f"{torch.abs(value_output - values).mean().item():.4f}"
                })
        
        # Get averages
        val_loss /= batch_count
        val_policy_acc /= batch_count
        val_value_mae /= batch_count
        
        # Update scheduler
        if scheduler:
            # Use accuracy
            scheduler.step(1 - val_policy_acc)  # Invert to maximize
        
        # Track history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        policy_accuracy_history.append(val_policy_acc)
        value_mae_history.append(val_value_mae)
        
        # Show summary
        print(f"Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Policy Acc: {train_policy_acc:.4f}, Value MAE: {train_value_mae:.6f}")
        print(f"  Val Loss: {val_loss:.4f}, Policy Acc: {val_policy_acc:.4f}, Value MAE: {val_value_mae:.6f}")
        
        # Check improvement
        improved = False
        
        # Check loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improved = True
            
            # Save model
            checkpoint_path = os.path.join(model_save_path, f"chess_cnn_best_loss_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'policy_acc': val_policy_acc,
                'value_mae': val_value_mae
            }, checkpoint_path)
            
            print(f"  Saved model checkpoint for best loss to {checkpoint_path}")
        
        # Check accuracy
        if val_policy_acc > best_policy_acc:
            best_policy_acc = val_policy_acc
            improved = True
            
            # Save model
            checkpoint_path = os.path.join(model_save_path, f"chess_cnn_best_accuracy_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'policy_acc': val_policy_acc,
                'value_mae': val_value_mae
            }, checkpoint_path)
            
            print(f"  Saved model checkpoint for best accuracy to {checkpoint_path}")
        
        # Update patience
        if improved:
            patience_counter = 0
            # Save best
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'policy_acc': val_policy_acc,
                'value_mae': val_value_mae
            }, os.path.join(model_save_path, "chess_cnn_best.pt"))
        else:
            patience_counter += 1
            print(f"  No improvement in validation metrics. Patience: {patience_counter}/{early_stopping_patience}")
        
        # Check stopping
        if patience_counter >= early_stopping_patience and epoch >= 20:
            # Minimum epochs
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Get time
    train_time = time.time() - start_time
    hours, rem = divmod(train_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Save history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'policy_accuracy': policy_accuracy_history,
        'value_mae': value_mae_history,
        'training_time': train_time,
        'best_policy_accuracy': best_policy_acc,
        'best_val_loss': best_val_loss
    }
    
    with open(os.path.join(model_save_path, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    
    # Plot results
    plt.figure(figsize=(16, 12))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(policy_accuracy_history, label='Policy Accuracy')
    plt.title('Policy Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(value_mae_history, label='Value MAE')
    plt.title('Value Mean Absolute Error History')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.grid(True, alpha=0.3)
    
    # Plot LR
    if hasattr(scheduler, 'history'):
        plt.subplot(2, 2, 4)
        plt.plot(scheduler.history, label='Learning Rate')
        plt.title('Learning Rate History')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_path, 'training_curves.png'))
    plt.close()
    
    return history

class ChessDataset(torch.utils.data.Dataset):
    """Chess position dataset"""
    def __init__(self, positions, evaluations, moves, transform=None):
        """
        Args:
            positions: Board encodings
            evaluations: Position evaluations
            moves: Best moves
            transform: Optional transforms
        """
        self.positions = positions
        self.evaluations = evaluations
        self.moves = moves
        self.transform = transform
        
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        position = self.positions[idx]
        evaluation = self.evaluations[idx]
        move = self.moves[idx]
        
        if self.transform:
            position = self.transform(position)
        
        return position, evaluation, move

def main():
    """Main training function"""
    # Parse args
    parser = argparse.ArgumentParser(description='Train ULTIMATE chess CNN model')
    parser.add_argument('--data', type=str, default='./processed_data/processed_data_high_quality.pt', 
                        help='Path to high quality processed data')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_filters', type=int, default=128, help='Number of filters in convolutional layers')
    parser.add_argument('--num_blocks', type=int, default=10, help='Number of residual blocks')
    parser.add_argument('--lr', type=float, default=0.00075, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=12, help='Early stopping patience')
    parser.add_argument('--output_dir', type=str, default='./models/final', help='Output directory for models')
    parser.add_argument('--use_cpu', action='store_true', help='Force CPU usage even if GPU is available')
    parser.add_argument('--use_original_data', action='store_true', 
                        help='Use original data instead of high quality data')
    
    args = parser.parse_args()
    
    # Force CPU
    global device
    if args.use_cpu:
        device = torch.device("cpu")
        print("Forcing CPU usage as requested")
    
    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Pick data
    data_path = './processed_data/processed_data.pt' if args.use_original_data else args.data
    
    # Check data
    if not os.path.exists(data_path):
        print(f"Processed data not found at {data_path}")
        if not args.use_original_data:
            print("High quality data not found. Trying to use original data instead...")
            data_path = './processed_data/processed_data.pt'
            if not os.path.exists(data_path):
                print("Original data not found either. Please run data_processing.py first")
                return
        else:
            print("Please run data_processing.py first")
            return
    
    # Make directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {data_path}")
    try:
        data = torch.load(data_path, map_location=torch.device('cpu'))  # Load to CPU
        positions = data['positions']
        evals = data['evals']
        moves = data['moves']
        
        # Show info
        print(f"Positions tensor shape: {positions.shape}, dtype: {positions.dtype}")
        print(f"Evals tensor shape: {evals.shape}, dtype: {evals.dtype}")
        print(f"Moves tensor shape: {moves.shape}, dtype: {moves.dtype}")
        
        # Fix types
        evals = evals.float()  # To float32
        moves = moves.long()   # To int64
        
        print(f"Loaded {len(positions)} positions")
        
        # Check distribution
        move_counts = torch.bincount(moves)
        most_common_moves = torch.argsort(move_counts, descending=True)[:10]
        print("Most common move indices and their counts:")
        for idx in most_common_moves:
            if move_counts[idx] > 0:  # Skip zeros
                print(f"Move index {idx}: {move_counts[idx]} occurrences ({move_counts[idx]/len(moves)*100:.2f}%)")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create dataset
    dataset = ChessDataset(positions, evals, moves)
    
    # Split data
    train_size = int(0.85 * len(dataset))  # More training
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True  # Faster GPU
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Prepared datasets: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    
    # Create model
    model = EnhancedChessCNN(
        input_channels=12,
        num_filters=args.num_filters, 
        num_residual_blocks=args.num_blocks,
        num_output_moves=1968  # Max moves
    ).to(device)
    
    print(f"Initialized enhanced model with {args.num_filters} filters and {args.num_blocks} residual blocks")
    
    # Loss functions
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning scheduler
    class CustomScheduler:
        def __init__(self, optimizer, factor=0.5, patience=5, min_lr=1e-6, verbose=True):
            self.optimizer = optimizer
            self.factor = factor
            self.patience = patience
            self.min_lr = min_lr
            self.verbose = verbose
            self.best_score = None
            self.counter = 0
            self.history = []
            
            # Get initial LR
            for param_group in self.optimizer.param_groups:
                self.history.append(param_group['lr'])
        
        def step(self, score):
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history.append(current_lr)
            
            if self.best_score is None:
                self.best_score = score
            elif score >= self.best_score:
                self.counter += 1
                if self.verbose:
                    print(f"LR scheduler: No improvement for {self.counter} epochs.")
                if self.counter >= self.patience:
                    self.counter = 0
                    new_lr = max(current_lr * self.factor, self.min_lr)
                    if new_lr != current_lr:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        if self.verbose:
                            print(f"Reducing learning rate from {current_lr} to {new_lr}")
            else:
                self.best_score = score
                self.counter = 0
    
    # Init scheduler
    scheduler = CustomScheduler(optimizer, factor=0.7, patience=3, min_lr=1e-6)
    
    # Train model
    try:
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            policy_criterion=policy_criterion,
            value_criterion=value_criterion,
            num_epochs=args.epochs,
            scheduler=scheduler,
            early_stopping_patience=args.patience,
            model_save_path=args.output_dir
        )
        
        print("Training complete!")
        print(f"Best validation loss: {history['best_val_loss']:.4f}")
        print(f"Best policy accuracy: {history['best_policy_accuracy']:.4f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        # Try CPU
        if device.type != 'cpu':
            print("Attempting to fall back to CPU training...")
            
            # Move to CPU
            model = model.to('cpu')
            
            # New optimizer
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=args.lr, 
                weight_decay=1e-4,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            scheduler = CustomScheduler(optimizer, factor=0.7, patience=3, min_lr=1e-6)
            
            # Try again
            try:
                history = train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    policy_criterion=policy_criterion,
                    value_criterion=value_criterion,
                    num_epochs=args.epochs,
                    scheduler=scheduler,
                    early_stopping_patience=args.patience,
                    model_save_path=args.output_dir
                )
                
                print("Training complete with CPU fallback!")
                print(f"Best validation loss: {history['best_val_loss']:.4f}")
                print(f"Best policy accuracy: {history['best_policy_accuracy']:.4f}")
                
            except Exception as cpu_e:
                print(f"Error during CPU training: {cpu_e}")

if __name__ == "__main__":
    main()