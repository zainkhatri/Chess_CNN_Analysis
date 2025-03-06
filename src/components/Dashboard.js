import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Chess } from 'chess.js';
import { Chessboard } from 'react-chessboard';
import './Dashboard.css';

// Stockfish Web Worker
const STOCKFISH_PATH = 'https://cdnjs.cloudflare.com/ajax/libs/stockfish.js/10.0.2/stockfish.js';

// Header Component
function Header() {
  return (
    <header className="app-header">
      <div className="header-content">
        <div className="logo">
          <Link to="/">Chess Analysis</Link>
        </div>
        <nav className="main-nav">
          <ul>
            <li><Link to="/">Home</Link></li>
            <li><Link to="/play">Play</Link></li>
            <li><Link to="/analysis">Analysis</Link></li>
          </ul>
        </nav>
      </div>
    </header>
  );
}

// Footer Component
function Footer() {
  return (
    <footer className="app-footer">
      <div className="footer-content">
        <p>&copy; {new Date().getFullYear()} Chess Post-Game Analysis App</p>
        <p>Powered by Neural Networks and Stockfish</p>
      </div>
    </footer>
  );
}

// Heatmap Component
function Heatmap({ data, squareSize = 40 }) {
  // Function to get color based on value (0 to 1)
  const getColor = (value) => {
    // Use a blue color scale from light to dark
    const r = Math.floor(255 * (1 - value));
    const g = Math.floor(255 * (1 - value));
    const b = 255;
    return `rgba(${r}, ${g}, ${b}, ${Math.min(0.8, value + 0.2)})`;
  };

  return (
    <div className="heatmap" style={{ width: squareSize * 8, height: squareSize * 8 }}>
      {data.map((row, rowIndex) => (
        <div key={rowIndex} className="heatmap-row" style={{ height: squareSize }}>
          {row.map((value, colIndex) => (
            <div
              key={colIndex}
              className="heatmap-cell"
              style={{
                width: squareSize,
                height: squareSize,
                backgroundColor: getColor(value),
              }}
            />
          ))}
        </div>
      ))}
    </div>
  );
}

// Chess Board Component with Stockfish Integration
function ChessBoard() {
  const [game, setGame] = useState(new Chess());
  const [fen, setFen] = useState('');
  const [playerColor, setPlayerColor] = useState('white');
  const [difficulty, setDifficulty] = useState(1200);
  const [gameStatus, setGameStatus] = useState('Choose your color and difficulty to start');
  const [isThinking, setIsThinking] = useState(false);
  const [moveHistory, setMoveHistory] = useState([]);
  const [gameStarted, setGameStarted] = useState(false);
  const [stockfish, setStockfish] = useState(null);
  const [engineReady, setEngineReady] = useState(false);
  const [selectedSquare, setSelectedSquare] = useState(null);
  const [possibleMoves, setPossibleMoves] = useState([]);
  const [bestMoves, setBestMoves] = useState([]); // Top engine moves
  const [evaluation, setEvaluation] = useState(0); // Current position evaluation
  
  const gameRef = useRef(new Chess());
  const engineDepth = useRef(difficulty < 1200 ? 8 : difficulty < 1500 ? 12 : 15);

  // Initialize Stockfish
  useEffect(() => {
    const worker = new Worker(STOCKFISH_PATH);
    
    worker.onmessage = (e) => {
      const message = e.data;
      
      // If Stockfish is ready
      if (message.includes('uciok')) {
        setEngineReady(true);
        worker.postMessage('setoption name MultiPV value 3'); // Show top 3 moves
        worker.postMessage('isready');
      }
      
      // Parse best move information
      if (message.includes('bestmove')) {
        const bestMove = message.split('bestmove ')[1].split(' ')[0];
        if (bestMove !== '(none)' && isThinking) {
          makeMove({ from: bestMove.slice(0, 2), to: bestMove.slice(2, 4), promotion: bestMove.length > 4 ? bestMove[4] : 'q' });
          setIsThinking(false);
        }
      }
      
      // Parse info line for evaluation
      if (message.includes('info') && message.includes('depth') && message.includes('score')) {
        // Only process completed depth
        const depthMatch = message.match(/depth (\d+)/);
        const currentDepth = depthMatch ? parseInt(depthMatch[1]) : 0;
        
        if (currentDepth === engineDepth.current && message.includes('pv')) {
          // Extract evaluation score
          let score = 0;
          let isMate = false;
          
          if (message.includes('score cp')) {
            const match = message.match(/score cp ([-\d]+)/);
            score = match ? parseInt(match[1]) / 100 : 0; // Convert centipawns to pawns
          } else if (message.includes('score mate')) {
            const match = message.match(/score mate ([-\d]+)/);
            score = match ? parseInt(match[1]) : 0;
            isMate = true;
          }
          
          // Adjust score for black's perspective
          if (gameRef.current.turn() === 'b') {
            score = -score;
          }
          
          // Extract move sequence
          const pvMatch = message.match(/pv ([a-h][1-8][a-h][1-8][qrbnk]? ?)+/);
          const moves = pvMatch ? pvMatch[0].split(' ').slice(1) : [];
          
          // Update best moves if this is a multipv line
          const multipvMatch = message.match(/multipv (\d+)/);
          if (multipvMatch) {
            const multipvIndex = parseInt(multipvMatch[1]) - 1;
            setBestMoves(prev => {
              const newMoves = [...prev];
              const moveInfo = {
                move: moves[0],
                score: score,
                isMate,
                line: moves.slice(0, 4).join(' ')
              };
              
              if (multipvIndex === 0) {
                setEvaluation(score);
              }
              
              // Replace at index or append
              newMoves[multipvIndex] = moveInfo;
              return newMoves.slice(0, 3); // Keep only top 3
            });
          }
        }
      }
    };
    
    // Initialize the engine
    worker.postMessage('uci');
    setStockfish(worker);
    
    // Clean up the worker when the component unmounts
    return () => {
      worker.terminate();
    };
  }, []);

  // Update the game state
  const updateGameState = useCallback(() => {
    const currentGame = gameRef.current;
    setFen(currentGame.fen());
    
    // Clear possible moves
    setPossibleMoves([]);
    setSelectedSquare(null);
    
    // Check game status
    if (currentGame.isGameOver()) {
      let status = '';
      if (currentGame.isCheckmate()) {
        status = `Checkmate! ${currentGame.turn() === 'w' ? 'Black' : 'White'} wins!`;
      } else if (currentGame.isDraw()) {
        status = 'Draw!';
        if (currentGame.isStalemate()) {
          status = 'Draw by stalemate!';
        } else if (currentGame.isThreefoldRepetition()) {
          status = 'Draw by threefold repetition!';
        } else if (currentGame.isInsufficientMaterial()) {
          status = 'Draw by insufficient material!';
        }
      }
      setGameStatus(status);
    } else {
      setGameStatus(`${currentGame.turn() === 'w' ? 'White' : 'Black'} to move`);
      
      // If it's the bot's turn, get its move
      if (gameStarted && 
          ((playerColor === 'white' && currentGame.turn() === 'b') || 
           (playerColor === 'black' && currentGame.turn() === 'w'))) {
        getBotMove();
      }
    }
    
    // Update move history
    const moves = [];
    const history = currentGame.history({ verbose: true });
    for (let i = 0; i < history.length; i += 2) {
      const moveNumber = Math.floor(i / 2) + 1;
      const whiteMove = history[i] ? `${history[i].san}` : '';
      const blackMove = history[i + 1] ? `${history[i + 1].san}` : '';
      moves.push({ moveNumber, whiteMove, blackMove });
    }
    setMoveHistory(moves);
    
    // Start engine analysis of position
    if (stockfish && engineReady && !currentGame.isGameOver()) {
      setBestMoves([]);
      setEvaluation(0);
      stockfish.postMessage(`position fen ${currentGame.fen()}`);
      stockfish.postMessage(`go depth ${engineDepth.current}`);
    }
  }, [engineReady, gameStarted, playerColor]);

  useEffect(() => {
    updateGameState();
  }, [updateGameState]);

  // Make a move on the board
  const makeMove = (move) => {
    const currentGame = gameRef.current;
    try {
      const result = currentGame.move(move);
      if (result) {
        setGame(new Chess(currentGame.fen()));
        updateGameState();
        return true;
      }
    } catch (error) {
      console.error('Invalid move:', error);
    }
    return false;
  };

  // Handle piece selection to show possible moves
  const onSquareClick = (square) => {
    if (!gameStarted || isThinking) return;
    
    const currentGame = gameRef.current;
    
    // Check if it's the player's turn
    if ((playerColor === 'white' && currentGame.turn() === 'b') || 
        (playerColor === 'black' && currentGame.turn() === 'w')) {
      return;
    }
    
    // If square is already selected, try to make a move
    if (selectedSquare && square !== selectedSquare) {
      const moveResult = makeMove({
        from: selectedSquare,
        to: square,
        promotion: 'q' // Always promote to queen for simplicity
      });
      
      if (moveResult) {
        setSelectedSquare(null);
        setPossibleMoves([]);
        return;
      }
    }
    
    // Select the square and show possible moves
    const piece = currentGame.get(square);
    if (piece && 
        ((piece.color === 'w' && playerColor === 'white') || 
         (piece.color === 'b' && playerColor === 'black'))) {
      setSelectedSquare(square);
      
      // Get all legal moves from this square
      const moves = currentGame.moves({
        square,
        verbose: true
      });
      
      setPossibleMoves(moves.map(move => move.to));
    } else {
      setSelectedSquare(null);
      setPossibleMoves([]);
    }
  };

  // Handle the player's move with drag and drop
  const onDrop = (sourceSquare, targetSquare) => {
    if (isThinking || !gameStarted) return false;
    
    // Check if it's the player's turn
    const currentGame = gameRef.current;
    if ((playerColor === 'white' && currentGame.turn() === 'b') || 
        (playerColor === 'black' && currentGame.turn() === 'w')) {
      return false;
    }
    
    // Try to make the move
    const move = {
      from: sourceSquare,
      to: targetSquare,
      promotion: 'q' // Always promote to queen for simplicity
    };
    
    return makeMove(move);
  };

  // Get bot move using Stockfish
  const getBotMove = () => {
    if (!stockfish || !engineReady) return;
    
    setIsThinking(true);
    
    // Use the current FEN position
    stockfish.postMessage(`position fen ${gameRef.current.fen()}`);
    
    // Adjust search time/depth based on difficulty
    const moveTime = difficulty === 800 ? 500 : 
                     difficulty === 1200 ? 1000 : 2000;
    
    // Add a small amount of randomness to easier levels
    if (difficulty <= 1200) {
      stockfish.postMessage(`setoption name Skill Level value ${difficulty === 800 ? 5 : 10}`);
    } else {
      stockfish.postMessage(`setoption name Skill Level value 20`);
    }
    
    // Ask for the best move
    stockfish.postMessage(`go movetime ${moveTime}`);
  };

  // Start a new game
  const startNewGame = () => {
    gameRef.current = new Chess();
    setGame(new Chess());
    setFen(gameRef.current.fen());
    setGameStatus('');
    setMoveHistory([]);
    setBestMoves([]);
    setEvaluation(0);
    setPossibleMoves([]);
    setSelectedSquare(null);
    setGameStarted(true);
    
    // Update engine depth based on difficulty
    engineDepth.current = difficulty < 1200 ? 8 : difficulty < 1500 ? 12 : 15;
    
    // If player is black, get the bot's first move
    if (playerColor === 'black') {
      getBotMove();
    }
  };

  // Reset the game
  const resetGame = () => {
    setGameStarted(false);
    gameRef.current = new Chess();
    setGame(new Chess());
    setFen(gameRef.current.fen());
    setGameStatus('Choose your color and difficulty to start');
    setMoveHistory([]);
    setBestMoves([]);
    setEvaluation(0);
    setPossibleMoves([]);
    setSelectedSquare(null);
  };

  // Format evaluation display
  const formatEvaluation = (evalScore, isMate = false) => {
    if (isMate) {
      return evalScore > 0 ? `Mate in ${evalScore}` : `Mate in ${-evalScore}`;
    }
    return evalScore > 0 ? `+${evalScore.toFixed(1)}` : evalScore.toFixed(1);
  };

  // Custom pieces with highlights for possible moves
  const customSquareStyles = {};
  
  // Add styles for selected square
  if (selectedSquare) {
    customSquareStyles[selectedSquare] = {
      backgroundColor: 'rgba(255, 255, 0, 0.4)',
    };
  }
  
  // Add styles for possible move destinations
  possibleMoves.forEach(square => {
    customSquareStyles[square] = {
      backgroundColor: 'rgba(0, 255, 0, 0.3)',
      borderRadius: '50%',
    };
  });

  return (
    <div className="chess-game-container">
      <div className="game-controls">
        <div className="control-panel">
          <h2>Play Chess</h2>
          {!gameStarted ? (
            <>
              <div className="config-options">
                <div className="option-group">
                  <label>Play as:</label>
                  <div className="button-group">
                    <button 
                      className={playerColor === 'white' ? 'active' : ''}
                      onClick={() => setPlayerColor('white')}
                    >
                      White
                    </button>
                    <button 
                      className={playerColor === 'black' ? 'active' : ''}
                      onClick={() => setPlayerColor('black')}
                    >
                      Black
                    </button>
                  </div>
                </div>
                
                <div className="option-group">
                  <label>Difficulty:</label>
                  <div className="button-group">
                    <button 
                      className={difficulty === 800 ? 'active' : ''}
                      onClick={() => setDifficulty(800)}
                    >
                      Beginner (800)
                    </button>
                    <button 
                      className={difficulty === 1200 ? 'active' : ''}
                      onClick={() => setDifficulty(1200)}
                    >
                      Intermediate (1200)
                    </button>
                    <button 
                      className={difficulty === 1500 ? 'active' : ''}
                      onClick={() => setDifficulty(1500)}
                    >
                      Advanced (1500)
                    </button>
                  </div>
                </div>
              </div>
              
              <button className="primary-button" onClick={startNewGame}>
                Start Game
              </button>
            </>
          ) : (
            <>
              <div className="game-status">
                <p>{gameStatus}</p>
                {isThinking && <p className="thinking">Computer is thinking...</p>}
                <p className="evaluation">Evaluation: {formatEvaluation(evaluation)}</p>
              </div>
              
              <div className="best-moves">
                <h4>Top Engine Moves:</h4>
                <ul>
                  {bestMoves.map((moveInfo, idx) => (
                    <li key={idx}>
                      <span className="move-notation">{moveInfo.move}</span>
                      <span className="move-eval">{formatEvaluation(moveInfo.score, moveInfo.isMate)}</span>
                    </li>
                  ))}
                </ul>
              </div>
              
              <div className="game-buttons">
                <button className="secondary-button" onClick={resetGame}>
                  Reset Game
                </button>
              </div>
            </>
          )}
        </div>
        
        {gameStarted && (
          <div className="move-history">
            <h3>Move History</h3>
            <div className="move-list">
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>White</th>
                    <th>Black</th>
                  </tr>
                </thead>
                <tbody>
                  {moveHistory.map((move) => (
                    <tr key={move.moveNumber}>
                      <td>{move.moveNumber}.</td>
                      <td>{move.whiteMove}</td>
                      <td>{move.blackMove}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
      
      <div className="board-container">
        <Chessboard
          id="main-board"
          position={fen}
          onPieceDrop={onDrop}
          onSquareClick={onSquareClick}
          boardOrientation={playerColor}
          customSquareStyles={customSquareStyles}
          customBoardStyle={{
            borderRadius: '4px',
            boxShadow: '0 4px 10px rgba(0, 0, 0, 0.5)'
          }}
          arePremovesAllowed={false}
          animationDuration={200}
          boardWidth={560}
        />
      </div>
    </div>
  );
}

// Main Dashboard Component
function Dashboard() {
  const navigate = useNavigate();

  return (
    <div className="app">
      <Header />
      
      <main className="main-content">
        {/* Landing/Home Page Content */}
        <div className="dashboard-container">
          <div className="dashboard-hero">
            <h1>Chess Post-Game Analysis & Move Prediction</h1>
            <p className="hero-subtitle">
              Play against bots of different skill levels and get detailed post-game analysis to improve your chess skills.
            </p>
            <button 
              className="cta-button" 
              onClick={() => navigate('/play')}
            >
              Start Playing
            </button>
          </div>
          
          <div className="features-section">
            <h2>Key Features</h2>
            <div className="features-grid">
              <div className="feature-card">
                <div className="feature-icon">🎮</div>
                <h3>Interactive Chess Experience</h3>
                <p>Play full chess games against bots of three different difficulty levels (800, 1200, and 1500 Elo).</p>
              </div>
              
              <div className="feature-card">
                <div className="feature-icon">🔍</div>
                <h3>Detailed Post-Game Analysis</h3>
                <p>Get a comprehensive breakdown of your game with error identification, move recommendations, and positional insights.</p>
              </div>
              
              <div className="feature-card">
                <div className="feature-icon">📊</div>
                <h3>Visual Heatmaps & Analytics</h3>
                <p>View visual representations of move strength, position control, and key turning points in the game.</p>
              </div>
              
              <div className="feature-card">
                <div className="feature-icon">🧠</div>
                <h3>AI-Powered Evaluations</h3>
                <p>Our custom neural network works alongside Stockfish to provide detailed evaluations and move suggestions.</p>
              </div>
              
              <div className="feature-card">
                <div className="feature-icon">📈</div>
                <h3>Progress Tracking</h3>
                <p>Learn from your mistakes and track your improvement over time with game statistics and analysis history.</p>
              </div>
            </div>
          </div>
          
          <div className="how-it-works">
            <h2>How It Works</h2>
            <div className="steps-container">
              <div className="step">
                <div className="step-number">1</div>
                <div className="step-content">
                  <h3>Play a Game</h3>
                  <p>Choose your preferred color and difficulty level, then play a complete chess game against our AI opponent.</p>
                </div>
              </div>
              
              <div className="step">
                <div className="step-number">2</div>
                <div className="step-content">
                  <h3>Analyze Your Game</h3>
                  <p>After the game, click "Analyze Game" to process your moves through our neural network and Stockfish engine.</p>
                </div>
              </div>
              
              <div className="step">
                <div className="step-number">3</div>
                <div className="step-content">
                  <h3>Review Insights</h3>
                  <p>Explore the detailed breakdown of your game, including move quality, positional insights, and improvement suggestions.</p>
                </div>
              </div>
              
              <div className="step">
                <div className="step-number">4</div>
                <div className="step-content">
                  <h3>Improve Your Skills</h3>
                  <p>Use the insights to understand your strengths and weaknesses, then apply what you've learned in your next game.</p>
                </div>
              </div>
            </div>
          </div>
          
          <div className="cta-section">
            <h2>Ready to improve your chess skills?</h2>
            <button 
              className="cta-button" 
              onClick={() => navigate('/play')}
            >
              Start Playing Now
            </button>
          </div>
          
          {/* Chess Board Component */}
          <ChessBoard />
        </div>
      </main>
      
      <Footer />
    </div>
  );
}

export default Dashboard;