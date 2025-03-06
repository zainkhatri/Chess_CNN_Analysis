import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import ChessCNN

# Check device availability - fall back to CPU if issues with MPS
device = torch.device("cpu")  # Default to CPU for stability
print(f"Default device: CPU")

# We'll try MPS if available, but with a fallback mechanism
try:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU")
except Exception as e:
    print(f"Error initializing GPU: {e}. Using CPU instead.")

def train_model(model, train_loader, val_loader, optimizer, policy_criterion, value_criterion, 
                num_epochs=20, scheduler=None, early_stopping_patience=5, model_save_path='./models'):
    """Train the chess CNN model"""
    # Create directory for model checkpoints
    os.makedirs(model_save_path, exist_ok=True)
    
    # Initialize variables for tracking training
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    policy_accuracy_history = []
    value_mae_history = []
    
    # Start training
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_policy_acc = 0
        train_value_mae = 0
        batch_count = 0
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for positions, values, moves in train_pbar:
            # Move data to device and ensure correct shapes
            positions = positions.to(device)
            values = values.to(device).float()  # Ensure float type
            moves = moves.to(device).long()  # Ensure long (int64) type
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            policy_output, value_output = model(positions)
            
            # Reshape value output if needed to match target
            if value_output.shape != values.shape:
                # Assuming value_output is [batch_size, 1] and values might be just [batch_size]
                if len(values.shape) == 1:
                    values = values.unsqueeze(1)  # Make it [batch_size, 1]
                # Or if value_output is [batch_size] and values is [batch_size, 1]
                elif len(value_output.shape) == 1:
                    value_output = value_output.unsqueeze(1)  # Make it [batch_size, 1]
            
            # Calculate loss
            policy_loss = policy_criterion(policy_output, moves)
            value_loss = value_criterion(value_output, values)
            
            # Combined loss (with weighting)
            loss = 0.7 * policy_loss + 0.3 * value_loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            
            # Calculate policy accuracy
            _, predicted = torch.max(policy_output, 1)
            train_policy_acc += (predicted == moves).sum().item() / moves.size(0)
            
            # Calculate value mean absolute error
            train_value_mae += torch.abs(value_output - values).mean().item()
            
            batch_count += 1
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'policy_acc': f"{(predicted == moves).sum().item() / moves.size(0):.4f}",
                'value_mae': f"{torch.abs(value_output - values).mean().item():.4f}"
            })
        
        # Calculate average training metrics
        train_loss /= batch_count
        train_policy_acc /= batch_count
        train_value_mae /= batch_count
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_policy_acc = 0
        val_value_mae = 0
        batch_count = 0
        
        # Progress bar for validation
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        
        with torch.no_grad():
            for positions, values, moves in val_pbar:
                # Move data to device and ensure correct types
                positions = positions.to(device)
                values = values.to(device).float()  # Ensure float type
                moves = moves.to(device).long()  # Ensure long (int64) type
                
                # Forward pass
                policy_output, value_output = model(positions)
                
                # Reshape value output if needed
                if value_output.shape != values.shape:
                    if len(values.shape) == 1:
                        values = values.unsqueeze(1)
                    elif len(value_output.shape) == 1:
                        value_output = value_output.unsqueeze(1)
                
                # Calculate loss
                policy_loss = policy_criterion(policy_output, moves)
                value_loss = value_criterion(value_output, values)
                loss = 0.7 * policy_loss + 0.3 * value_loss
                
                # Update metrics
                val_loss += loss.item()
                
                # Calculate policy accuracy
                _, predicted = torch.max(policy_output, 1)
                val_policy_acc += (predicted == moves).sum().item() / moves.size(0)
                
                # Calculate value mean absolute error
                val_value_mae += torch.abs(value_output - values).mean().item()
                
                batch_count += 1
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}", 
                    'policy_acc': f"{(predicted == moves).sum().item() / moves.size(0):.4f}",
                    'value_mae': f"{torch.abs(value_output - values).mean().item():.4f}"
                })
        
        # Calculate average validation metrics
        val_loss /= batch_count
        val_policy_acc /= batch_count
        val_value_mae /= batch_count
        
        # Update learning rate scheduler if provided
        if scheduler:
            scheduler.step(val_loss)
        
        # Track metrics history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        policy_accuracy_history.append(val_policy_acc)
        value_mae_history.append(val_value_mae)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Policy Acc: {train_policy_acc:.4f}, Value MAE: {train_value_mae:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Policy Acc: {val_policy_acc:.4f}, Value MAE: {val_value_mae:.4f}")
        
        # Save model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save model checkpoint
            checkpoint_path = os.path.join(model_save_path, f"chess_cnn_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'policy_acc': val_policy_acc,
                'value_mae': val_value_mae
            }, checkpoint_path)
            
            print(f"  Saved model checkpoint to {checkpoint_path}")
            
            # Also save the best model
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
            print(f"  No improvement in validation loss. Patience: {patience_counter}/{early_stopping_patience}")
        
        # Check early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Calculate training time
    train_time = time.time() - start_time
    hours, rem = divmod(train_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'policy_accuracy': policy_accuracy_history,
        'value_mae': value_mae_history,
        'training_time': train_time
    }
    
    with open(os.path.join(model_save_path, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    
    # Plot training curves
    plt.figure(figsize=(15, 10))
    
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
    
    plt.subplot(2, 2, 3)
    plt.plot(value_mae_history, label='Value MAE')
    plt.title('Value Mean Absolute Error History')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_path, 'training_curves.png'))
    plt.close()
    
    return history

class ChessDataset(torch.utils.data.Dataset):
    """Dataset for chess positions"""
    def __init__(self, positions, evaluations, moves, transform=None):
        """
        Args:
            positions: List of encoded board positions (12 channels)
            evaluations: Corresponding stockfish evaluations
            moves: Corresponding best moves (encoded as indices)
            transform: Optional transform to apply
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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train chess CNN model')
    parser.add_argument('--data', type=str, default='./processed_data/processed_data.pt', help='Path to processed data')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_filters', type=int, default=64, help='Number of filters in convolutional layers')
    parser.add_argument('--num_blocks', type=int, default=6, help='Number of residual blocks')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--output_dir', type=str, default='./models', help='Output directory for models')
    parser.add_argument('--use_cpu', action='store_true', help='Force CPU usage even if GPU is available')
    
    args = parser.parse_args()
    
    # Force CPU if requested
    global device
    if args.use_cpu:
        device = torch.device("cpu")
        print("Forcing CPU usage as requested")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if processed data exists
    if not os.path.exists(args.data):
        print(f"Processed data not found at {args.data}")
        print("Please run data_processing.py first")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load processed data
    print(f"Loading data from {args.data}")
    try:
        data = torch.load(args.data, map_location=torch.device('cpu'))  # Always load to CPU first
        positions = data['positions']
        evals = data['evals']
        moves = data['moves']
        
        # Debug info about tensors
        print(f"Positions tensor shape: {positions.shape}, dtype: {positions.dtype}")
        print(f"Evals tensor shape: {evals.shape}, dtype: {evals.dtype}")
        print(f"Moves tensor shape: {moves.shape}, dtype: {moves.dtype}")
        
        # Explicitly convert dtypes if needed
        evals = evals.float()  # Ensure float32
        moves = moves.long()   # Ensure int64 (long)
        
        print(f"Loaded {len(positions)} positions")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create dataset
    dataset = ChessDataset(positions, evals, moves)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders with fewer workers for Mac
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Prepared datasets: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    
    # Initialize model with smaller size for M3 Mac
    model = ChessCNN(
        input_channels=12,
        num_filters=args.num_filters,  # Reduced from 128 to 64
        num_residual_blocks=args.num_blocks,  # Reduced from 10 to 6
        num_output_moves=1968  # Maximum number of move possibilities
    ).to(device)
    
    print(f"Initialized model with {args.num_filters} filters and {args.num_blocks} residual blocks")
    
    # Define loss functions
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    # Define optimizer with weight decay to prevent overfitting
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Train the model
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
    except Exception as e:
        print(f"Error during training: {e}")
        # If we hit GPU errors, try falling back to CPU
        if device.type != 'cpu':
            print("Attempting to fall back to CPU training...")
            
            # Move model to CPU
            model = model.to('cpu')
            
            # Reinitialize optimizer
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
            
            # Try training again
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
            except Exception as cpu_e:
                print(f"Error during CPU training: {cpu_e}")

if __name__ == "__main__":
    main()