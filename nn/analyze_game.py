import os
import sys
import json
import argparse
import chess
import chess.pgn
from io import StringIO
from game_advisor import ChessAdvisor

def save_analysis_for_frontend(moves, analysis, output_file):
    """Save game analysis in a format suitable for the React frontend"""
    # Format for frontend
    frontend_data = {
        "positions": [],
        "summary": {
            "white_accuracy": 0,
            "black_accuracy": 0,
            "white_mistakes": 0,
            "white_blunders": 0,
            "black_mistakes": 0,
            "black_blunders": 0,
            "decisive_moments": []
        },
        "metadata": {
            "white": "Player",
            "black": "Computer",
            "date": "2025-03-04",
            "result": "Unknown"
        }
    }
    
    # Process positions
    for i, pos_data in enumerate(analysis):
        position = pos_data['position']
        
        # Skip initial
        if i > 0:
            move_number = pos_data['move_number']
            
            # Get SAN notation
            board = chess.Board(analysis[i-1]['position']['fen'])
            move = chess.Move.from_uci(pos_data['move'])
            san_move = board.san(move)
            
            side = 'white' if move_number % 2 == 1 else 'black'
            
            # Check quality
            quality = 'best'
            eval_loss = 0
            
            if i > 0 and position['stockfish_eval'] is not None and analysis[i-1]['position']['stockfish_eval'] is not None:
                prev_eval = analysis[i-1]['position']['stockfish_eval']
                curr_eval = position['stockfish_eval']
                
                # Calculate loss
                if side == 'white':
                    eval_loss = (prev_eval - curr_eval) / 100  # Convert to pawns
                else:
                    eval_loss = (curr_eval - prev_eval) / 100  # Convert to pawns
                
                # Label moves
                if eval_loss >= 2.0:
                    quality = 'blunder'
                    if side == 'white':
                        frontend_data['summary']['white_blunders'] += 1
                    else:
                        frontend_data['summary']['black_blunders'] += 1
                elif eval_loss >= 1.0:
                    quality = 'mistake'
                    if side == 'white':
                        frontend_data['summary']['white_mistakes'] += 1
                    else:
                        frontend_data['summary']['black_mistakes'] += 1
                elif eval_loss >= 0.5:
                    quality = 'inaccuracy'
                
                # Track key moments
                if eval_loss >= 1.0:
                    frontend_data['summary']['decisive_moments'].append({
                        'move_number': move_number,
                        'quality': quality,
                        'centipawn_loss': eval_loss * 100
                    })
        
        # Add position data
        frontend_position = {
            'fen': position['fen'],
            'move_number': pos_data['move_number'],
            'analysis': {
                'stockfish_eval': position['stockfish_eval'],
                'model_eval': position['model_eval'],
                'heatmap': {
                    'heatmap': position['attention_map']
                },
                'top_moves': position['stockfish_moves'],
                'material_balance': {
                    'white_material': 0,  # Need calculation
                    'black_material': 0,  # Need calculation
                    'balance': 0
                },
                'mobility': {
                    'white_total': 0,  # Need calculation
                    'black_total': 0   # Need calculation
                }
            }
        }
        
        # Add move quality
        if i > 0:
            frontend_position['evaluation'] = {
                'quality': quality,
                'centipawn_loss': eval_loss * 100
            }
        
        frontend_data['positions'].append(frontend_position)
    
    # Calculate accuracy
    white_moves = sum(1 for pos in frontend_data['positions'] if 'evaluation' in pos and pos['move_number'] % 2 == 1)
    black_moves = sum(1 for pos in frontend_data['positions'] if 'evaluation' in pos and pos['move_number'] % 2 == 0)
    
    white_good_moves = sum(1 for pos in frontend_data['positions'] 
                         if 'evaluation' in pos and pos['move_number'] % 2 == 1 and pos['evaluation']['quality'] != 'blunder' and pos['evaluation']['quality'] != 'mistake')
    
    black_good_moves = sum(1 for pos in frontend_data['positions'] 
                         if 'evaluation' in pos and pos['move_number'] % 2 == 0 and pos['evaluation']['quality'] != 'blunder' and pos['evaluation']['quality'] != 'mistake')
    
    if white_moves > 0:
        frontend_data['summary']['white_accuracy'] = 100 * white_good_moves / white_moves
    
    if black_moves > 0:
        frontend_data['summary']['black_accuracy'] = 100 * black_good_moves / black_moves
    
    # Sort key moments
    frontend_data['summary']['decisive_moments'].sort(key=lambda x: x['centipawn_loss'], reverse=True)
    
    # Save result
    with open(output_file, 'w') as f:
        json.dump(frontend_data, f, indent=2)
    
    return frontend_data

def main():
    parser = argparse.ArgumentParser(description='Analyze a chess game and prepare data for the frontend')
    parser.add_argument('--pgn', type=str, help='PGN file to analyze')
    parser.add_argument('--moves', type=str, help='JSON file with move history')
    parser.add_argument('--output', type=str, default='analysis_results.json', help='Output file for analysis results')
    parser.add_argument('--model', type=str, default='./models/chess_cnn_best.pt', help='Path to trained model')
    
    args = parser.parse_args()
    
    if not args.pgn and not args.moves:
        print("Error: Either --pgn or --moves must be specified")
        parser.print_help()
        sys.exit(1)
    
    # Setup advisor
    advisor = ChessAdvisor(model_path=args.model, stockfish_path="stockfish")
    
    # Process input
    game_results = None
    
    if args.pgn:
        # Read PGN
        with open(args.pgn, 'r') as f:
            pgn_content = f.read()
        
        # Analyze game
        game_results = advisor.analyze_pgn(pgn_content)
    
    elif args.moves:
        # Read moves
        with open(args.moves, 'r') as f:
            moves_data = json.load(f)
        
        # Extract moves
        if isinstance(moves_data, list):
            moves = moves_data
        elif isinstance(moves_data, dict) and 'moves' in moves_data:
            moves = moves_data['moves']
        else:
            print("Error: Could not find moves array in JSON file")
            sys.exit(1)
        
        # Analyze game
        game_analysis = advisor.analyze_game(moves)
        
        # Create results
        game_results = {
            'headers': {
                'white': "Player",
                'black': "Computer",
                'date': "2025-03-04",
                'result': "Unknown"
            },
            'moves': moves,
            'analysis': game_analysis
        }
    
    if not game_results:
        print("Error: Failed to analyze game")
        sys.exit(1)
    
    # Format for frontend
    frontend_data = save_analysis_for_frontend(game_results['moves'], game_results['analysis'], args.output)
    
    # Create chart
    advisor.visualize_evaluation(game_results['analysis'], output_file='evaluation_chart.png')
    
    print(f"Analysis complete! Results saved to {args.output}")
    print(f"Evaluation chart saved to evaluation_chart.png")
    
    # Show summary
    print("\nGame Summary:")
    print(f"White Accuracy: {frontend_data['summary']['white_accuracy']:.1f}%")
    print(f"Black Accuracy: {frontend_data['summary']['black_accuracy']:.1f}%")
    print(f"White Mistakes: {frontend_data['summary']['white_mistakes']}, Blunders: {frontend_data['summary']['white_blunders']}")
    print(f"Black Mistakes: {frontend_data['summary']['black_mistakes']}, Blunders: {frontend_data['summary']['black_blunders']}")
    
    if frontend_data['summary']['decisive_moments']:
        print("\nKey Moments:")
        for moment in frontend_data['summary']['decisive_moments'][:3]:
            print(f"Move {moment['move_number']}: {moment['quality'].capitalize()} (-{moment['centipawn_loss']:.1f} centipawns)")

if __name__ == "__main__":
    main()