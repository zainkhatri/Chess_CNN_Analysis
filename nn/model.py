import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    """Residual block with identity mapping"""
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

class AttentionModule(nn.Module):
    """Spatial attention module to focus on important board areas"""
    def __init__(self, channels):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels // 4, kernel_size=1)
        self.conv2 = nn.Conv2d(channels // 4, 1, kernel_size=1)
        self.attention_map = None
        
    def forward(self, x):
        # Generate attention map
        attention = F.relu(self.conv1(x))
        attention = torch.sigmoid(self.conv2(attention))
        
        # Apply attention
        out = x * attention
        
        # Store attention map for visualization
        self.attention_map = attention
        
        return out

class ChessCNN(nn.Module):
    """Chess position evaluation and move prediction CNN"""
    def __init__(self, input_channels=12, num_filters=128, num_residual_blocks=10, num_output_moves=1968):
        """
        Args:
            input_channels: Number of input channels (12 for 6 piece types x 2 colors)
            num_filters: Number of filters in convolutional layers
            num_residual_blocks: Number of residual blocks in the network
            num_output_moves: Number of possible moves (up to 1968 legal moves in chess)
        """
        super(ChessCNN, self).__init__()
        
        # Initial convolution block
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
        
        # Policy head (move prediction)
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, num_output_moves)
        )
        
        # Value head (position evaluation)
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()  # Output between -1 and 1 (scaling to centipawns can be done externally)
        )
        
    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Apply attention
        x = self.attention(x)
        
        # Policy and value heads
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value
    
    def get_attention_map(self):
        """Returns the most recent attention map for visualization"""
        if self.attention.attention_map is None:
            return None
        return self.attention.attention_map.squeeze().detach().cpu().numpy()


def encode_board(fen):
    """Convert FEN string to 12-channel 8x8 representation
    Channels: [WPawn, WKnight, WBishop, WRook, WQueen, WKing, BPawn, BKnight, BBishop, BRook, BQueen, BKing]
    """
    # Extract the board part of the FEN
    board_fen = fen.split(' ')[0]
    
    # Initialize 12 channel representation (6 piece types x 2 colors)
    board_tensor = np.zeros((12, 8, 8), dtype=np.float32)
    
    # Piece mapping: lowercase is black, uppercase is white
    piece_to_channel = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    
    # Parse the FEN and fill the tensor
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


def move_to_index(move, board_size=8):
    """Convert a move string (e.g., 'e2e4') to an index for the policy head"""
    # Decode algebraic notation
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
    
    # Basic encoding: (from_square * 64 + to_square) * 5 + promotion
    return (from_idx * 64 + to_idx) * 5 + promotion


def index_to_move(index, board_size=8):
    """Convert an index back to a move string (e.g., 'e2e4')"""
    # Decode the index
    promotion_idx = index % 5
    index = index // 5
    to_idx = index % 64
    from_idx = index // 64
    
    # Calculate row and column
    from_row, from_col = from_idx // board_size, from_idx % board_size
    to_row, to_col = to_idx // board_size, to_idx % board_size
    
    # Convert to algebraic notation
    from_square = chr(from_col + ord('a')) + str(from_row + 1)
    to_square = chr(to_col + ord('a')) + str(to_row + 1)
    
    # Add promotion piece if applicable
    promotion_pieces = ['', 'n', 'b', 'r', 'q']
    promotion = promotion_pieces[promotion_idx] if promotion_idx < len(promotion_pieces) else ''
    
    return from_square + to_square + promotion


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