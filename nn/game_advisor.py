import os
import json
import torch
import chess
import chess.pgn
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

from model import ChessCNN, encode_board, index_to_move

class ChessAdvisor:
    """Chess game advisor that analyzes games and provides feedback"""
    
    def __init__(self, model_path=None, stockfish_path=None):
        """Initialize the chess advisor"""
        # Check device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA GPU")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")
        
        # Load model
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                self.model = ChessCNN(
                    input_channels=12,
                    num_filters=64,
                    num_residual_blocks=6,
                    num_output_moves=1968
                ).to(self.device)
                
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
        
        # Init Stockfish
        self.stockfish = None
        if stockfish_path:
            try:
                import chess.engine
                self.stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                self.stockfish.configure({"Threads": 2})
                print("Stockfish engine initialized")
            except Exception as e:
                print(f"Error initializing Stockfish: {e}")
                self.stockfish = None
    
    def __del__(self):
        """Clean up resources"""
        if self.stockfish:
            self.stockfish.quit()
    
    def analyze_position(self, fen, depth=10):
        """Analyze a single position"""
        # Setup board
        board = chess.Board(fen)
        
        # Model analysis
        model_eval = None
        model_moves = []
        attention_map = None
        
        if self.model:
            try:
                # Encode position
                encoded_board = encode_board(fen).unsqueeze(0).to(self.device)
                
                # Get prediction
                with torch.no_grad():
                    policy_output, value_output = self.model(encoded_board)
                    
                    # Get attention
                    attention_map = self.model.get_attention_map()
                    
                    # Get evaluation
                    model_eval = value_output.item() * 100  # To centipawns
                    
                    # Get move probabilities
                    policy_probs = torch.softmax(policy_output, dim=1)[0]
                    
                    # Filter legal moves
                    legal_moves = list(board.legal_moves)
                    legal_move_scores = []
                    
                    for i, prob in enumerate(policy_probs):
                        try:
                            move_str = index_to_move(i)
                            move = chess.Move.from_uci(move_str)
                            
                            if move in legal_moves:
                                legal_move_scores.append({
                                    "uci": move_str,
                                    "san": board.san(move),
                                    "probability": float(prob.item())
                                })
                        except:
                            continue
                    
                    # Sort moves
                    legal_move_scores.sort(key=lambda x: x["probability"], reverse=True)
                    
                    # Top moves
                    model_moves = legal_move_scores[:3] if legal_move_scores else []
            except Exception as e:
                print(f"Error in model evaluation: {e}")
        
        # Stockfish analysis
        stockfish_eval = None
        stockfish_moves = []
        
        if self.stockfish and board.is_valid():
            try:
                result = self.stockfish.analyse(board, chess.engine.Limit(depth=depth), multipv=3)
                
                for i, pv in enumerate(result):
                    # Get score
                    score = pv["score"].white().score(mate_score=10000)
                    if i == 0:
                        stockfish_eval = score
                    
                    # Get move
                    move = pv["pv"][0]
                    stockfish_moves.append({
                        "uci": move.uci(),
                        "san": board.san(move),
                        "score": score
                    })
            except Exception as e:
                print(f"Error in stockfish evaluation: {e}")
        
        return {
            'fen': fen,
            'model_eval': model_eval,
            'model_moves': model_moves,
            'stockfish_eval': stockfish_eval,
            'stockfish_moves': stockfish_moves,
            'attention_map': attention_map.tolist() if attention_map is not None else None
        }
    
    def analyze_game(self, moves, starting_fen=chess.STARTING_FEN):
        """Analyze a game from a list of moves"""
        board = chess.Board(starting_fen)
        analysis = []
        
        # Starting position
        analysis.append({
            'move_number': 0,
            'move': None,
            'position': self.analyze_position(board.fen())
        })
        
        # Analyze moves
        for i, move_str in enumerate(moves):
            try:
                move = board.parse_uci(move_str)
                board.push(move)
                
                analysis.append({
                    'move_number': i + 1,
                    'move': move_str,
                    'position': self.analyze_position(board.fen())
                })
            except Exception as e:
                print(f"Error processing move {move_str}: {e}")
                break
        
        return analysis
    
    def analyze_pgn(self, pgn_str):
        """Analyze a game from PGN string"""
        pgn = StringIO(pgn_str)
        game = chess.pgn.read_game(pgn)
        
        if not game:
            return None
        
        # Get metadata
        headers = {
            'white': game.headers.get("White", "Unknown"),
            'black': game.headers.get("Black", "Unknown"),
            'date': game.headers.get("Date", "Unknown"),
            'result': game.headers.get("Result", "Unknown"),
            'white_elo': game.headers.get("WhiteElo", "?"),
            'black_elo': game.headers.get("BlackElo", "?")
        }
        
        # Get moves
        moves = []
        board = game.board()
        for move in game.mainline_moves():
            moves.append(move.uci())
            board.push(move)
        
        # Full analysis
        analysis = self.analyze_game(moves)
        
        return {
            'headers': headers,
            'moves': moves,
            'analysis': analysis
        }
    
    def find_mistakes(self, game_analysis, threshold=50):
        """Find mistakes in the analyzed game
        
        Args:
            game_analysis: Game analysis from analyze_game or analyze_pgn
            threshold: Centipawn threshold for considering a move a mistake
            
        Returns:
            List of mistakes with their position and suggested corrections
        """
        mistakes = []
        
        for i in range(1, len(game_analysis)):
            prev_pos = game_analysis[i-1]['position']
            curr_pos = game_analysis[i]['position']
            
            # Need evaluations
            if prev_pos['stockfish_eval'] is None or curr_pos['stockfish_eval'] is None:
                continue
            
            # Get evals
            prev_eval = prev_pos['stockfish_eval']
            curr_eval = curr_pos['stockfish_eval']
            
            # Adjust for side
            move_num = game_analysis[i]['move_number']
            side_to_move = 'white' if move_num % 2 == 1 else 'black'
            
            # Calculate loss
            eval_diff = prev_eval - curr_eval if side_to_move == 'white' else curr_eval - prev_eval
            
            # Check threshold
            if eval_diff >= threshold:
                # Get best move
                best_move = None
                if prev_pos['stockfish_moves']:
                    best_move = prev_pos['stockfish_moves'][0]
                
                # Add mistake
                mistakes.append({
                    'move_number': move_num,
                    'side': side_to_move,
                    'played_move': game_analysis[i]['move'],
                    'best_move': best_move,
                    'eval_loss': eval_diff,
                    'current_position': curr_pos['fen'],
                    'previous_position': prev_pos['fen'],
                    'mistake_severity': 'blunder' if eval_diff >= 200 else 'mistake' if eval_diff >= 100 else 'inaccuracy'
                })
        
        return mistakes
    
    def visualize_evaluation(self, game_analysis, output_file=None):
        """Visualize evaluation changes throughout the game"""
        move_numbers = []
        model_evals = []
        stockfish_evals = []
        
        for move_data in game_analysis:
            move_numbers.append(move_data['move_number'])
            
            if move_data['position']['model_eval'] is not None:
                model_evals.append(move_data['position']['model_eval'] / 100)  # To pawns
            else:
                model_evals.append(None)
                
            if move_data['position']['stockfish_eval'] is not None:
                stockfish_evals.append(move_data['position']['stockfish_eval'] / 100)  # To pawns
            else:
                stockfish_evals.append(None)
        
        # Filter nulls
        valid_indices = [i for i, x in enumerate(stockfish_evals) if x is not None]
        valid_move_numbers = [move_numbers[i] for i in valid_indices]
        valid_stockfish_evals = [stockfish_evals[i] for i in valid_indices]
        
        valid_model_indices = [i for i, x in enumerate(model_evals) if x is not None]
        valid_model_move_numbers = [move_numbers[i] for i in valid_model_indices]
        valid_model_evals = [model_evals[i] for i in valid_model_indices]
        
        plt.figure(figsize=(12, 6))
        
        # Plot Stockfish
        if valid_stockfish_evals:
            plt.plot(valid_move_numbers, valid_stockfish_evals, 'r-', label='Stockfish')
        
        # Plot CNN
        if valid_model_evals:
            plt.plot(valid_model_move_numbers, valid_model_evals, 'b-', label='Neural Network')
        
        # Zero line
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Mark sides
        for i in range(len(move_numbers)):
            if i % 2 == 0:  # White's move
                plt.axvspan(i-0.5, i+0.5, alpha=0.1, color='lightgray')
        
        # Make pretty
        plt.title('Position Evaluation Throughout Game')
        plt.xlabel('Move Number')
        plt.ylabel('Evaluation (pawns)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add note
        plt.figtext(0.5, 0.01, 'Positive values favor White, negative values favor Black', 
                    ha='center', fontsize=10, style='italic')
        
        if output_file:
            plt.savefig(output_file, dpi=100, bbox_inches='tight')
            plt.close()
            return output_file
        else:
            return plt
    
    def get_positional_advice(self, game_analysis, move_number):
        """Get specific advice for a position"""
        # Find position
        position_data = None
        for move_data in game_analysis:
            if move_data['move_number'] == move_number:
                position_data = move_data
                break
        
        if not position_data:
            return None
        
        position = position_data['position']
        side_to_move = 'white' if move_number % 2 == 1 else 'black'
        
        # Get candidate moves
        stockfish_moves = position['stockfish_moves']
        model_moves = position['model_moves']
        
        # Check agreement
        model_agrees = False
        if model_moves and stockfish_moves:
            if model_moves[0]['uci'] == stockfish_moves[0]['uci']:
                model_agrees = True
        
        # Create advice
        advice = {
            'move_number': move_number,
            'side_to_move': side_to_move,
            'fen': position['fen'],
            'stockfish_eval': position['stockfish_eval'] / 100 if position['stockfish_eval'] else None,
            'model_eval': position['model_eval'] / 100 if position['model_eval'] else None,
            'stockfish_moves': stockfish_moves,
            'model_moves': model_moves,
            'model_agrees_with_stockfish': model_agrees,
            'attention_map': position['attention_map']
        }
        
        return advice
    
    def generate_game_report(self, game_analysis, mistake_threshold=50):
        """Generate a comprehensive game report"""
        # Find mistakes
        mistakes = self.find_mistakes(game_analysis, threshold=mistake_threshold)
        
        # Count moves
        white_moves = sum(1 for move in game_analysis if move['move_number'] % 2 == 1 and move['move_number'] > 0)
        black_moves = sum(1 for move in game_analysis if move['move_number'] % 2 == 0 and move['move_number'] > 0)
        
        # Count errors
        white_mistakes = sum(1 for mistake in mistakes if mistake['side'] == 'white')
        black_mistakes = sum(1 for mistake in mistakes if mistake['side'] == 'black')
        
        # Calculate accuracy
        white_accuracy = 100 * (1 - white_mistakes / white_moves) if white_moves > 0 else 100
        black_accuracy = 100 * (1 - black_mistakes / black_moves) if black_moves > 0 else 100
        
        # Count types
        white_blunders = sum(1 for mistake in mistakes if mistake['side'] == 'white' and mistake['mistake_severity'] == 'blunder')
        white_mistakes_med = sum(1 for mistake in mistakes if mistake['side'] == 'white' and mistake['mistake_severity'] == 'mistake')
        white_inaccuracies = sum(1 for mistake in mistakes if mistake['side'] == 'white' and mistake['mistake_severity'] == 'inaccuracy')
        
        black_blunders = sum(1 for mistake in mistakes if mistake['side'] == 'black' and mistake['mistake_severity'] == 'blunder')
        black_mistakes_med = sum(1 for mistake in mistakes if mistake['side'] == 'black' and mistake['mistake_severity'] == 'mistake')
        black_inaccuracies = sum(1 for mistake in mistakes if mistake['side'] == 'black' and mistake['mistake_severity'] == 'inaccuracy')
        
        # Create report
        report = {
            'total_moves': len(game_analysis) - 1,  # Skip initial
            'white_moves': white_moves,
            'black_moves': black_moves,
            'white_stats': {
                'accuracy': white_accuracy,
                'mistakes': white_mistakes,
                'blunders': white_blunders,
                'medium_mistakes': white_mistakes_med,
                'inaccuracies': white_inaccuracies
            },
            'black_stats': {
                'accuracy': black_accuracy,
                'mistakes': black_mistakes,
                'blunders': black_blunders,
                'medium_mistakes': black_mistakes_med,
                'inaccuracies': black_inaccuracies
            },
            'mistakes': mistakes
        }
        
        return report

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python game_advisor.py <pgn_file>")
        sys.exit(1)
    
    # Init advisor
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'chess_cnn_best.pt')
    advisor = ChessAdvisor(model_path=model_path, stockfish_path="stockfish")
    
    # Load game
    with open(sys.argv[1], 'r') as f:
        pgn_content = f.read()
    
    # Analyze
    game_results = advisor.analyze_pgn(pgn_content)
    
    if not game_results:
        print("Failed to parse PGN file")
        sys.exit(1)
    
    # Generate report
    report = advisor.generate_game_report(game_results['analysis'])
    
    # Show summary
    print("\nGame Summary:")
    print("-------------")
    print(f"White: {game_results['headers']['white']} ({game_results['headers']['white_elo']})")
    print(f"Black: {game_results['headers']['black']} ({game_results['headers']['black_elo']})")
    print(f"Result: {game_results['headers']['result']}")
    print(f"Total Moves: {report['total_moves']}")
    print("\nWhite Performance:")
    print(f"  Accuracy: {report['white_stats']['accuracy']:.1f}%")
    print(f"  Blunders: {report['white_stats']['blunders']}")
    print(f"  Mistakes: {report['white_stats']['medium_mistakes']}")
    print(f"  Inaccuracies: {report['white_stats']['inaccuracies']}")
    print("\nBlack Performance:")
    print(f"  Accuracy: {report['black_stats']['accuracy']:.1f}%")
    print(f"  Blunders: {report['black_stats']['blunders']}")
    print(f"  Mistakes: {report['black_stats']['medium_mistakes']}")
    print(f"  Inaccuracies: {report['black_stats']['inaccuracies']}")
    
    # Show key moments
    if report['mistakes']:
        print("\nKey Moments:")
        for mistake in sorted(report['mistakes'], key=lambda x: x['eval_loss'], reverse=True)[:5]:
            print(f"  Move {mistake['move_number']} ({mistake['side'].capitalize()}): {mistake['mistake_severity'].capitalize()}")
            print(f"    Played: {mistake['played_move']}")
            if mistake['best_move']:
                print(f"    Better: {mistake['best_move']['san']} (saves {mistake['eval_loss']/100:.1f} pawns)")
    
    # Save evaluation chart
    plt = advisor.visualize_evaluation(game_results['analysis'])
    plt.savefig('game_evaluation.png')
    print("\nEvaluation chart saved to game_evaluation.png")