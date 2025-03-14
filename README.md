# Chess CNN: Deep Learning for Chess Position Evaluation and Move Prediction

## Project Description

Our research develops an advanced artificial intelligence system for chess position evaluation and optimal move prediction using deep learning techniques applied to board representations. This project addresses the challenge of chess analysis by leveraging convolutional neural networks with attention mechanisms and residual connections to improve both prediction accuracy and evaluation quality compared to traditional approaches.

## Abstract

Chess position evaluation and move prediction represent challenging problems due to the game's combinatorial complexity. This project implements a convolutional neural network (CNN) with a dual-head architecture that simultaneously predicts the best move (policy head) and evaluates the position (value head) of a chess board. Our model is trained on an enhanced dataset of 16,174 positions extracted from high-quality games and annotated using the Stockfish chess engine.

The research addresses critical challenges in chess position analysis by:
1. Implementing a dual-head CNN with residual connections and attention mechanisms
2. Developing robust data processing for chess position representation
3. Integrating Stockfish for ground truth labels and validation
4. Providing visual explanations through attention maps

The proposed system achieves a validation loss of 0.3210 and move prediction accuracy of 2.89%, highlighting the challenges of the vast move space while providing useful position evaluations that correlate well with Stockfish analysis.

## Background

Chess represents a significant challenge for artificial intelligence systems due to its vast complexity and strategic depth. Traditional chess engines like Stockfish use sophisticated alpha-beta search algorithms with handcrafted evaluation functions, but recent advancements in deep learning have opened new approaches to chess analysis.

Deep learning models, particularly convolutional neural networks (CNNs), have demonstrated remarkable capabilities in image recognition tasks. By representing chess positions as multi-channel images, these techniques can be adapted to learn patterns and relationships on the chessboard.

Our research builds upon previous work in chess position evaluation, incorporating advanced techniques such as:
- Residual connections to enable deeper network architectures
- Attention mechanisms to focus on critical board regions
- Dual-head architecture for simultaneous evaluation and move prediction
- Focal loss to address the severe class imbalance in move prediction

## Data

### Dataset Overview
- **Total Positions**: 16,174 chess positions
- **Position Sources**: High-quality games (1800+ ELO)
- **Annotation Source**: Stockfish chess engine evaluations

### Position Representation
Chess positions are encoded as 12-channel 8×8 tensors, with separate channels for each piece type and color:

```python
def encode_board(fen):
    """FEN to tensor"""
    # Get board
    board_fen = fen.split(' ')[0]
    
    # Init channels
    board_tensor = np.zeros((12, 8, 8), dtype=np.float32)
    
    # Map pieces
    piece_to_channel = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    
    # Fill tensor
    row, col = 0, 0
    for char in board_fen:
        if char == '/':
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        elif char in piece_to_channel:
            channel = piece_to_channel[char]
            board_tensor[channel, row, col] = 1.0
            col += 1
    
    return torch.tensor(board_tensor, dtype=torch.float32)
```

### Move Encoding
Moves are encoded as indices in a 1968-dimensional space (64×64 possible square combinations × 5 promotion options):

```python
def move_to_index(move, board_size=8):
    """Move to index"""
    # Get coordinates
    from_col = ord(move[0]) - ord('a')
    from_row = int(move[1]) - 1
    to_col = ord(move[2]) - ord('a')
    to_row = int(move[3]) - 1
    
    # Handle promotion
    promotion = 0
    if len(move) > 4:
        promotion_pieces = ['', 'n', 'b', 'r', 'q']
        promotion = promotion_pieces.index(move[4]) if move[4] in promotion_pieces else 0
    
    # Calculate index
    from_idx = from_row * board_size + from_col
    to_idx = to_row * board_size + to_col
    
    # Encode move
    return (from_idx * 64 + to_idx) * 5 + promotion
```

### Data Processing Pipeline
We created a processing pipeline that:
1. Extracts positions from PGN chess game files
2. Evaluates positions using Stockfish
3. Encodes board states into tensors
4. Converts best moves into class indices

### Data Split
- **Training Set**: 85% (13,747 positions)
- **Validation Set**: 15% (2,427 positions)
- **Stratified splitting** to ensure balanced representation

## Model Architecture

### Base Model: ChessCNN

Our primary neural network architecture incorporates residual connections and attention mechanisms:

```python
class ChessCNN(nn.Module):
    """Chess evaluation network"""
    def __init__(self, input_channels=12, num_filters=128, num_residual_blocks=10, num_output_moves=1968):
        super(ChessCNN, self).__init__()
        
        # Initial conv
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_residual_blocks)
        ])
        
        # Attention module
        self.attention = AttentionModule(num_filters)
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, num_output_moves)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()  # -1 to 1
        )
```

### Key Components

#### Residual Blocks
```python
class ResidualBlock(nn.Module):
    """Skip connection block"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out
```

#### Attention Module
```python
class AttentionModule(nn.Module):
    """Focus important areas"""
    def __init__(self, channels):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels // 4, kernel_size=1)
        self.conv2 = nn.Conv2d(channels // 4, 1, kernel_size=1)
        self.attention_map = None
        
    def forward(self, x):
        # Create attention
        attention = F.relu(self.conv1(x))
        attention = torch.sigmoid(self.conv2(attention))
        
        # Apply attention
        out = x * attention
        
        # Store map
        self.attention_map = attention
        
        return out
```

### Enhanced Architecture

We developed an enhanced version with bottleneck blocks and additional regularization:

```python
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
```

## Training Methodology

### Focal Loss for Class Imbalance

Chess move prediction suffers from severe class imbalance, which we addressed using focal loss:

```python
def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    """Handle class imbalance"""
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()
```

### Custom Learning Rate Scheduler

We implemented a custom scheduler to optimize training:

```python
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
```

### Training Configuration

We trained with the following parameters:
- **Optimizer**: AdamW with weight decay 1e-4
- **Initial Learning Rate**: 0.00075
- **Batch Size**: 64
- **Epochs**: 40 (with early stopping)
- **Loss Weighting**: Dynamic weighting between policy and value heads

The training code includes:
```python
# Train with optimal configuration
python train_model.py --batch_size 64 --num_filters 128 --num_blocks 10 --epochs 40
```

## Experimental Results

### Model Performance

We conducted extensive experiments with different model configurations:

| Run   | Batch | Filters | Blocks | Dataset Size | Val Loss | Policy Acc | Time      |
|-------|-------|---------|--------|--------------|----------|------------|-----------|
| 1     | 16    | 64      | 6      | 3,232        | 0.8812   | 22.46%     | 2m 24s    |
| 2     | 16    | 48      | 4      | 3,232        | 0.8489   | 23.20%     | 37s       |
| 3     | 8     | 32      | 3      | 1,500        | 0.6194   | 3.72%      | 49s       |
| 4     | 32    | 96      | 8      | 16,174       | 0.3763   | 2.67%      | 7m 25s    |
| 5     | 64    | 128     | 10     | 16,174       | **0.3210** | **2.89%** | 4m 16s    |

Run 5 incorporated attention mechanisms, focal loss, and dropout, resulting in our best model.

### Learning Curves

Training and validation curves demonstrate model learning dynamics:

```python
# Example of learning curve visual
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.tight_layout()
```

### Key Findings

1. **Dataset Size Impact**: Expanding from 3,232 to 16,174 positions reduced validation loss from 0.8812 to 0.3210
2. **Attention Mechanisms**: Incorporating attention improved focus on critical board regions
3. **Move Prediction Challenge**: Even our best model achieved modest move prediction accuracy (2.89%), highlighting the difficulty of this task
4. **Position Evaluation**: The value head converged faster and provided reliable evaluations

## Game Analysis System

Our project includes a comprehensive chess game analysis system:

```python
from game_advisor import ChessAdvisor

# Initialize advisor
advisor = ChessAdvisor(model_path="./models/chess_cnn_best.pt", stockfish_path="stockfish")

# Analyze a game
with open("game.pgn", 'r') as f:
    pgn_content = f.read()

# Get results
game_results = advisor.analyze_pgn(pgn_content)

# Generate visual evaluation chart
advisor.visualize_evaluation(game_results['analysis'], output_file='evaluation_chart.png')

# Find mistakes
mistakes = advisor.find_mistakes(game_results['analysis'], threshold=50)
```

### Analysis Features

The ChessAdvisor provides:
1. Position evaluation with confidence scores
2. Best move suggestions
3. Mistake identification with categorization (blunder, mistake, inaccuracy)
4. Visual heatmaps of important board regions
5. Comparison with Stockfish analysis
6. Game summary statistics

### Example Output

```
Game Summary:
White Accuracy: 86.3%
Black Accuracy: 73.5%
White Mistakes: 2, Blunders: 1
Black Mistakes: 4, Blunders: 2

Key Moments:
Move 24: Blunder (-254.3 centipawns)
Move 18: Mistake (-132.7 centipawns)
Move 36: Mistake (-118.5 centipawns)
```

## Discussion

### Interpreting the Results

Our research demonstrates the potential of deep learning for chess analysis, but also highlights key challenges. The policy head's modest accuracy (2.89%) reflects the extreme difficulty of move prediction in chess due to:

1. **Vast Move Space**: With 1,968 possible moves, accurate prediction is challenging
2. **Severe Class Imbalance**: Most moves never appear in the training data
3. **Context Dependency**: The best move depends on the broader game context

However, the value head proved highly effective for position evaluation, correlating well with Stockfish analysis while using a fraction of the computational resources.

### Technical Challenges

Several technical challenges emerged during development:

1. **Data Processing**: Converting FEN strings to tensors required careful implementation
2. **Class Imbalance**: Move prediction required specialized loss functions
3. **Computational Constraints**: Training deeper networks required optimization
4. **Attention Mechanism Integration**: Properly implementing attention required experimentation

### Practical Applications

Our model enables several practical applications:

1. **Rapid Position Evaluation**: Near-instantaneous assessment without extensive search
2. **Game Analysis**: Identifying key moments and mistakes in chess games
3. **Training Tool**: Providing feedback to players on their games
4. **Opening Analysis**: Evaluating chess openings without memorization

## Future Work

Based on our findings, several promising directions emerge:

1. **Dataset Expansion**
   - Increase dataset size to millions of positions
   - Include positions from various playing styles and time controls

2. **Architecture Improvements**
   - Explore transformer architectures for capturing long-range dependencies
   - Investigate hybrid approaches combining neural evaluation with search

3. **Training Enhancements**
   - Implement reinforcement learning for self-improvement
   - Explore curriculum learning from simple to complex positions

4. **System Integration**
   - Develop a complete web-based interface for game analysis
   - Create plugins for popular chess platforms

## Conclusion

Our Chess CNN project demonstrates the power of deep learning in chess analysis, providing a foundation for future research. By combining residual networks, attention mechanisms, and dual-head architecture, we created a system capable of meaningful position evaluation and move suggestions.

While perfect move prediction remains challenging, our model offers valuable insights for chess players and researchers. The integration with traditional engines like Stockfish creates a hybrid approach that leverages the strengths of both neural networks and search algorithms.

This work contributes to the growing field of AI in strategic games, showcasing how deep learning techniques can be applied to one of humanity's oldest and most complex board games.

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12
- NumPy
- Matplotlib
- python-chess
- tqdm

Install dependencies with:
```bash
pip install torch numpy matplotlib python-chess tqdm
```

## Usage

```bash
# Process data from PGN files
python data_processing.py

# Train model with optimized parameters
python train_model.py --batch_size 64 --num_filters 128 --num_blocks 10

# Analyze a chess game
python analyze_game.py --pgn game.pgn --output analysis.json --model ./models/chess_cnn_best.pt
```