import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class ChessCNN(nn.Module):
    """Chess evaluation network"""
    def __init__(self, input_channels=12, num_filters=128, num_residual_blocks=10, num_output_moves=1968):
        """
        Args:
            input_channels: Piece channels
            num_filters: Filter count
            num_residual_blocks: Block count
            num_output_moves: Possible moves
        """
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
        
    def forward(self, x):
        # Initial conv
        x = self.initial_conv(x)
        
        # Apply blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Apply attention
        x = self.attention(x)
        
        # Get outputs
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value
    
    def get_attention_map(self):
        """Return attention map"""
        if self.attention.attention_map is None:
            return None
        return self.attention.attention_map.squeeze().detach().cpu().numpy()


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


def index_to_move(index, board_size=8):
    """Index to move"""
    # Decode index
    promotion_idx = index % 5
    index = index // 5
    to_idx = index % 64
    from_idx = index // 64
    
    # Get coordinates
    from_row, from_col = from_idx // board_size, from_idx % board_size
    to_row, to_col = to_idx // board_size, to_idx % board_size
    
    # Get notation
    from_square = chr(from_col + ord('a')) + str(from_row + 1)
    to_square = chr(to_col + ord('a')) + str(to_row + 1)
    
    # Add promotion
    promotion_pieces = ['', 'n', 'b', 'r', 'q']
    promotion = promotion_pieces[promotion_idx] if promotion_idx < len(promotion_pieces) else ''
    
    return from_square + to_square + promotion


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