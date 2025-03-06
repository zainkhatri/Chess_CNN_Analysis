import React, { useState, useEffect, useRef } from 'react';
import { Chess } from 'chess.js';
import { Chessboard } from 'react-chessboard';
import './Board.css';

// Enhanced Stockfish loading function
function loadStockfish() {
  return new Promise((resolve, reject) => {
    try {
      // Try to create a web worker directly
      console.log("Attempting to load Stockfish as Web Worker...");
      try {
        const worker = new Worker('/stockfish.js');
        console.log("Successfully loaded Stockfish as Web Worker");
        resolve(worker);
        return;
      } catch (workerError) {
        console.error("Failed to load as Web Worker:", workerError);
      }

      // Try local file with script tag
      const script = document.createElement('script');
      script.src = '/stockfish.js';
      script.async = true;
      script.onerror = () => {
        console.log("Failed to load local Stockfish, trying CDN...");
        
        // First CDN fallback
        const cdnScript1 = document.createElement('script');
        cdnScript1.src = 'https://unpkg.com/stockfish@11.0.0/stockfish.js';
        cdnScript1.async = true;
        cdnScript1.onerror = () => {
          console.log("Failed first CDN, trying alternate CDN...");
          
          // Second CDN fallback
          const cdnScript2 = document.createElement('script');
          cdnScript2.src = 'https://cdnjs.cloudflare.com/ajax/libs/stockfish.js/10.0.2/stockfish.js';
          cdnScript2.async = true;
          cdnScript2.onerror = () => {
            console.error("Failed to load Stockfish from all sources");
            reject(new Error("Failed to load Stockfish from all sources"));
          };
          cdnScript2.onload = () => {
            console.log("Loaded Stockfish from alternate CDN");
            try {
              if (typeof window.STOCKFISH === 'function') {
                resolve(window.STOCKFISH());
              } else if (typeof window.Stockfish === 'function') {
                resolve(window.Stockfish());
              } else {
                const worker = new Worker(cdnScript2.src);
                resolve(worker);
              }
            } catch (err) {
              console.error("Error initializing engine:", err);
              reject(err);
            }
          };
          document.body.appendChild(cdnScript2);
        };
        
        cdnScript1.onload = () => {
          console.log("Loaded Stockfish from first CDN");
          try {
            if (typeof window.STOCKFISH === 'function') {
              resolve(window.STOCKFISH());
            } else if (typeof window.Stockfish === 'function') {
              resolve(window.Stockfish());
            } else {
              const worker = new Worker(cdnScript1.src);
              resolve(worker);
            }
          } catch (err) {
            console.error("Error initializing engine:", err);
            reject(err);
          }
        };
        document.body.appendChild(cdnScript1);
      };
      
      script.onload = () => {
        console.log("Loaded local Stockfish");
        try {
          if (typeof window.STOCKFISH === 'function') {
            resolve(window.STOCKFISH());
          } else if (typeof window.Stockfish === 'function') {
            resolve(window.Stockfish());
          } else {
            const worker = new Worker(script.src);
            resolve(worker);
          }
        } catch (err) {
          console.error("Error initializing engine:", err);
          reject(err);
        }
      };
      document.body.appendChild(script);
    } catch (error) {
      console.error("General error loading Stockfish:", error);
      reject(error);
    }
  });
}

function Board() {
  const [game, setGame] = useState(new Chess());
  const [fen, setFen] = useState('');
  const [playerColor, setPlayerColor] = useState('white');
  const [difficulty, setDifficulty] = useState(1200);
  const [gameStatus, setGameStatus] = useState('Choose your color and difficulty to start');
  const [isThinking, setIsThinking] = useState(false);
  const [moveHistory, setMoveHistory] = useState([]);
  const [gameStarted, setGameStarted] = useState(false);
  const [selectedSquare, setSelectedSquare] = useState(null);
  const [possibleMoves, setPossibleMoves] = useState([]);
  const [evaluation, setEvaluation] = useState(0);
  const [bestMoves, setBestMoves] = useState([]);
  const [analysisMode, setAnalysisMode] = useState(false);
  const [currentMoveIndex, setCurrentMoveIndex] = useState(0);
  const [fullMoveHistory, setFullMoveHistory] = useState([]);
  const [engineLoaded, setEngineLoaded] = useState(false);
  const [engineError, setEngineError] = useState(null);
  
  const gameRef = useRef(new Chess());
  const engineRef = useRef(null);
  const waitingForMove = useRef(false);
  
  // Load Stockfish with improved error handling
  useEffect(() => {
    async function setupEngine() {
      try {
        console.log("Setting up Stockfish engine...");
        const engine = await loadStockfish();
        engineRef.current = engine;
        
        // Enhanced engine message handler
        engine.onmessage = (event) => {
          const message = event.data;
          handleEngineMessage(message);
        };
        
        // Improved engine initialization with better parameters
        engine.postMessage('uci');
        engine.postMessage('setoption name MultiPV value 3');
        engine.postMessage('setoption name Threads value 4'); // Use more threads if available
        engine.postMessage('setoption name Hash value 256');  // Increase hash size for better analysis
        engine.postMessage('isready');
        
        setEngineLoaded(true);
        setEngineError(null);
        console.log("Stockfish engine loaded successfully");
      } catch (error) {
        console.error("Failed to load Stockfish:", error);
        setEngineError("Failed to load Stockfish engine. Using simplified bot.");
        setEngineLoaded(false);
      }
    }
    
    setupEngine();
    
    return () => {
      if (engineRef.current) {
        engineRef.current.postMessage('quit');
      }
    };
  }, []);
  
  // Enhanced engine message handler
  const handleEngineMessage = (message) => {
    // Log all engine messages for debugging
    console.log("Engine message:", message);
    
    // Enhanced bestmove handling
    if (message.includes('bestmove') && waitingForMove.current) {
      waitingForMove.current = false;
      const messageParts = message.split(' ');
      const bestMoveIndex = messageParts.indexOf('bestmove');
      
      if (bestMoveIndex !== -1 && bestMoveIndex + 1 < messageParts.length) {
        const bestMove = messageParts[bestMoveIndex + 1];
        
        if (bestMove && bestMove !== '(none)') {
          // Verify the move is legal before making it
          const currentFen = gameRef.current.fen();
          const testGame = new Chess(currentFen);
          try {
            const moveObj = {
              from: bestMove.slice(0, 2),
              to: bestMove.slice(2, 4),
              promotion: bestMove.length > 4 ? bestMove[4] : 'q'
            };
            
            console.log("Engine wants to make move:", moveObj);
            
            // Validate move with the test game
            const validMove = testGame.move(moveObj);
            
            if (validMove) {
              console.log("Move is valid, making move:", moveObj);
              makeMove(moveObj);
            } else {
              console.error("Engine suggested invalid move:", moveObj);
              // Fall back to a random legal move
              const legalMoves = gameRef.current.moves({ verbose: true });
              if (legalMoves.length > 0) {
                const randomMove = legalMoves[Math.floor(Math.random() * legalMoves.length)];
                console.log("Making random move instead:", randomMove);
                makeMove({
                  from: randomMove.from,
                  to: randomMove.to,
                  promotion: randomMove.promotion || 'q'
                });
              }
            }
          } catch (error) {
            console.error("Error validating engine move:", error);
            getImprovedFallbackMove(); // Fall back to the improved AI
          }
        }
      }
      
      setIsThinking(false);
    }
    
    // Enhanced evaluation parsing
    if (message.includes('info') && message.includes('depth') && message.includes('score')) {
      try {
        // Extract depth to ensure we only process deep enough analysis
        const depthMatch = message.match(/depth (\d+)/);
        const depth = depthMatch ? parseInt(depthMatch[1]) : 0;
        
        // Only process deep enough analysis
        if (depth < 12) return;
        
        // Extract score with improved parsing
        let score = 0;
        let isMate = false;
        
        if (message.includes('score cp')) {
          const match = message.match(/score cp ([-\d]+)/);
          score = match ? parseInt(match[1], 10) / 100 : 0; // Convert centipawns to pawns
        } else if (message.includes('score mate')) {
          const match = message.match(/score mate ([-\d]+)/);
          score = match ? parseInt(match[1], 10) : 0;
          isMate = true;
        }
        
        // Adjust score for black's perspective
        if (gameRef.current.turn() === 'b') {
          score = -score;
        }
        
        // Extract move sequence with better regex
        const pvMatch = message.match(/pv ((?:[a-h][1-8][a-h][1-8][qrbnk]? ?)+)/);
        if (!pvMatch || pvMatch.length < 2) return;
        
        const moves = pvMatch[1].trim().split(' ');
        if (moves.length === 0) return;
        
        // Update best moves if this is a multipv line
        const multipvMatch = message.match(/multipv (\d+)/);
        if (multipvMatch) {
          const multipvIndex = parseInt(multipvMatch[1], 10) - 1;
          
          setBestMoves(prev => {
            const newMoves = [...prev];
            
            // Try to convert UCI moves to SAN notation for better readability
            let moveNotation = moves[0];
            try {
              // Make a temporary game to convert UCI to SAN
              const tempGame = new Chess(gameRef.current.fen());
              const move = {
                from: moveNotation.substring(0, 2),
                to: moveNotation.substring(2, 4),
                promotion: moveNotation.length > 4 ? moveNotation[4] : undefined
              };
              const result = tempGame.move(move);
              if (result) {
                moveNotation = result.san;
              }
            } catch (error) {
              // Keep UCI notation if conversion fails
              console.warn("Failed to convert UCI to SAN:", error);
            }
            
            const moveInfo = {
              uci: moves[0],
              san: moveNotation,
              score: score,
              isMate,
              line: moves.slice(0, 5).join(' ') // Show more of the line
            };
            
            if (multipvIndex === 0) {
              setEvaluation(score);
            }
            
            newMoves[multipvIndex] = moveInfo;
            return newMoves.slice(0, 3); // Keep only top 3
          });
        }
      } catch (error) {
        console.error("Error parsing engine message:", error, message);
      }
    }
  };

  // Update the game state
  const updateGameState = () => {
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
      
      // If it's the bot's turn and we're not in analysis mode, get its move
      if (gameStarted && !analysisMode &&
          ((playerColor === 'white' && currentGame.turn() === 'b') || 
           (playerColor === 'black' && currentGame.turn() === 'w'))) {
        setTimeout(() => getBotMove(), 300); // Reduced delay for better UX
      }
    }
    
    // Update move history
    const moves = [];
    const history = currentGame.history({ verbose: true });
    setFullMoveHistory(history);
    
    for (let i = 0; i < history.length; i += 2) {
      const moveNumber = Math.floor(i / 2) + 1;
      const whiteMove = history[i] ? history[i].san : '';
      const blackMove = history[i + 1] ? history[i + 1].san : '';
      moves.push({ moveNumber, whiteMove, blackMove });
    }
    setMoveHistory(moves);
    
    // Start engine analysis of position if engine is ready
    if (engineRef.current && engineLoaded && !currentGame.isGameOver() && !waitingForMove.current) {
      setBestMoves([]);
      setEvaluation(0);
      
      // Send the current position to the engine
      engineRef.current.postMessage('ucinewgame'); // Reset engine state for clean analysis
      engineRef.current.postMessage('position fen ' + currentGame.fen());
      engineRef.current.postMessage('go depth 15');
    }
  };

  useEffect(() => {
    updateGameState();
  }, []);

  // Make a move on the board
  const makeMove = (move) => {
    const currentGame = gameRef.current;
    try {
      console.log("Attempting to make move:", move);
      const result = currentGame.move(move);
      if (result) {
        console.log("Move made successfully:", result);
        setGame(new Chess(currentGame.fen()));
        updateGameState();
        return true;
      } else {
        console.error("Move rejected by chess.js:", move);
      }
    } catch (error) {
      console.error('Invalid move error:', error, move);
    }
    return false;
  };

  // Function to map ELO to appropriate Stockfish skill level
  const mapEloToSkillLevel = (elo) => {
    // Stockfish's skill levels 0-20 approximately map from 800-2800 ELO
    if (elo <= 800) return 0;
    if (elo >= 2800) return 20;
    return Math.round((elo - 800) / 100);
  };

  // Improved bot move calculation using Stockfish
  const getBotMove = () => {
    if (!engineRef.current || !engineLoaded) {
      console.log("Engine not ready, using fallback AI");
      getImprovedFallbackMove();
      return;
    }
    
    try {
      setIsThinking(true);
      waitingForMove.current = true;
      
      // Debug info
      console.log("Current board FEN:", gameRef.current.fen());
      console.log("Legal moves:", gameRef.current.moves({verbose: true}));
      
      // Force Stockfish to calculate a better ELO-appropriate move
      let depth, moveTime;
      
      switch (difficulty) {
        case 800: // Beginner
          depth = 8;
          moveTime = 1000;
          break;
        case 1200: // Intermediate
          depth = 12;
          moveTime = 1500;
          break;
        case 1500: // Advanced
          depth = 16;
          moveTime = 2000;
          break;
        case 2200: // Expert
          depth = 20;
          moveTime = 3000;
          break;
        default:
          depth = 12;
          moveTime = 1500;
      }
      
      // Reset engine for clean calculation
      engineRef.current.postMessage('ucinewgame');
      
      // Set appropriate skill level based on ELO
      const skillLevel = mapEloToSkillLevel(difficulty);
      engineRef.current.postMessage(`setoption name Skill Level value ${skillLevel}`);
      
      // Configure engine strength more accurately
      if (difficulty <= 1500) {
        // Use UCI_LimitStrength for more accurate ELO play
        engineRef.current.postMessage(`setoption name UCI_LimitStrength value true`);
        engineRef.current.postMessage(`setoption name UCI_Elo value ${difficulty}`);
      } else {
        // For higher ELOs, let engine calculate more deeply
        engineRef.current.postMessage(`setoption name UCI_LimitStrength value false`);
      }
      
      // Set contempt based on ELO - higher rated engines should play more aggressively
      const contempt = Math.min(Math.floor(difficulty / 100), 50);
      engineRef.current.postMessage(`setoption name Contempt value ${contempt}`);
      
      // Set search depth limits based on ELO
      const maxDepth = Math.min(depth, difficulty < 1500 ? 14 : 20);
      
      // For lower difficulties, use MultiPV to enable selecting non-optimal moves
      if (difficulty < 1500) {
        engineRef.current.postMessage(`setoption name MultiPV value 3`);
      } else {
        engineRef.current.postMessage(`setoption name MultiPV value 1`);
      }
      
      // If lower difficulty, sometimes deliberately restrict search depth
      if (difficulty <= 1200 && Math.random() < 0.3) {
        depth = Math.max(6, depth - 4); // Reduce depth for occasional weaker moves
      }
      
      // Send current position - use FEN instead of move history
      engineRef.current.postMessage('position fen ' + gameRef.current.fen());
      
      // Get the best move with both depth and time constraint
      engineRef.current.postMessage(`go depth ${maxDepth} movetime ${moveTime}`);
      
    } catch (error) {
      console.error("Error using Stockfish:", error);
      getImprovedFallbackMove();
    }
  };

  // Improved fallback AI move function with better heuristics
  const getImprovedFallbackMove = () => {
    setIsThinking(true);
    
    setTimeout(() => {
      const currentGame = gameRef.current;
      const legalMoves = currentGame.moves({ verbose: true });
      
      console.log("Fallback AI considering moves:", legalMoves);
      
      if (legalMoves.length === 0) {
        setIsThinking(false);
        return;
      }
      
      // Group moves into categories with better strategic evaluation
      const checkmateMoves = []; // Moves that deliver checkmate
      const captureMoves = []; // Capturing moves
      const checkMoves = []; // Check-giving moves
      const centerMoves = []; // Center control moves
      const developingMoves = []; // Piece development moves
      const castlingMoves = []; // Castling moves
      const escapeMoves = []; // Moves that escape check or capture
      const otherMoves = []; // All other moves
      
      // Evaluate board position to get material count
      const materialCount = getMaterialCount(currentGame);
      
      // Calculate if we're in endgame
      const isEndgame = (materialCount.white + materialCount.black) < 20;
      
      // Go through all legal moves and categorize them
      for (const move of legalMoves) {
        // Create a temporary game to evaluate the move
        const tempGame = new Chess(currentGame.fen());
        tempGame.move(move);
        
        // Check if move delivers checkmate
        if (tempGame.isCheckmate()) {
          checkmateMoves.push({ move, score: 1000 });
          continue;
        }
        
        // Get base score for the move
        let moveScore = 0;
        
        // Add score for captures based on piece values
        if (move.flags.includes('c')) {
          const capturedPieceValue = getPieceValue(move.captured);
          const movingPieceValue = getPieceValue(move.piece);
          
          // MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)
          // Prioritize capturing high-value pieces with low-value pieces
          moveScore += capturedPieceValue * 10 - movingPieceValue;
          captureMoves.push({ move, score: moveScore });
        }
        
        // Add score for check-giving moves
        if (tempGame.isCheck()) {
          moveScore += 5;
          checkMoves.push({ move, score: moveScore });
        }
        
        // Add score for center control (e4, d4, e5, d5)
        const centerSquares = ['e4', 'd4', 'e5', 'd5'];
        if (centerSquares.includes(move.to)) {
          moveScore += 3;
          centerMoves.push({ move, score: moveScore });
        }
        
        // Add score for castling - highly valuable in opening/middlegame
        if (move.flags.includes('k') || move.flags.includes('q')) {
          moveScore += 8;
          castlingMoves.push({ move, score: moveScore });
        }
        
        // Add score for development (moving pieces out in opening)
        const piecesStillHome = countPiecesInHomeRow(currentGame);
        if (!isEndgame && piecesStillHome > 2 && !['p', 'k'].includes(move.piece)) {
          // Piece is not a pawn or king, and we're not in endgame
          const fromRank = move.from.charAt(1);
          const toRank = move.to.charAt(1);
          
          // Moving pieces off the back rank in opening
          if ((currentGame.turn() === 'w' && fromRank === '1' && toRank !== '1') || 
              (currentGame.turn() === 'b' && fromRank === '8' && toRank !== '8')) {
            moveScore += 2;
            developingMoves.push({ move, score: moveScore });
          }
        }
        
        // Add score for moves that escape from being captured
        // Check if the current piece is under attack
        const fromSquare = move.from;
        const isUnderAttack = isSquareAttacked(currentGame, fromSquare);
        if (isUnderAttack) {
          moveScore += getPieceValue(move.piece) * 2; // More valuable pieces get higher escape priority
          escapeMoves.push({ move, score: moveScore });
        }
        
        // If move doesn't fit a specific category, it goes to otherMoves
        if (moveScore === 0) {
          otherMoves.push({ move, score: 0 });
        }
      }
      
      // Choose a move based on difficulty
      let selectedMove;
      
      // If there's a checkmate move, always take it
      if (checkmateMoves.length > 0) {
        selectedMove = checkmateMoves[0].move;
      } else {
        let moveProbabilities = [];
        
        // Completely different weights for different ELO levels
        if (difficulty >= 2200) {
          // Expert - strongly prefers the best tactical moves with few mistakes
          moveProbabilities = [
            { category: checkmateMoves, weight: 100 },
            { category: escapeMoves, weight: 95 },
            { category: captureMoves, weight: 90 },
            { category: checkMoves, weight: 85 },
            { category: castlingMoves, weight: 80 },
            { category: centerMoves, weight: 75 },
            { category: developingMoves, weight: 70 },
            { category: otherMoves, weight: 5 }
          ];
        } 
        else if (difficulty >= 1500) {
          // Advanced - prefers good moves but makes occasional mistakes
          moveProbabilities = [
            { category: checkmateMoves, weight: 100 },
            { category: escapeMoves, weight: 85 },
            { category: captureMoves, weight: 75 },
            { category: checkMoves, weight: 65 },
            { category: castlingMoves, weight: 60 },
            { category: centerMoves, weight: 50 },
            { category: developingMoves, weight: 40 },
            { category: otherMoves, weight: 10 }
          ];
        } 
        else if (difficulty >= 1200) {
          // Intermediate - makes some sound moves but also significant errors
          moveProbabilities = [
            { category: checkmateMoves, weight: 100 },
            { category: escapeMoves, weight: 70 },
            { category: captureMoves, weight: 60 },
            { category: checkMoves, weight: 50 },
            { category: castlingMoves, weight: 40 },
            { category: centerMoves, weight: 30 },
            { category: developingMoves, weight: 25 },
            { category: otherMoves, weight: 20 }
          ];
        } 
        else {
          // Beginner - mostly random with occasional good moves
          moveProbabilities = [
            { category: checkmateMoves, weight: 100 },
            { category: escapeMoves, weight: 60 }, // Still try to save pieces sometimes
            { category: captureMoves, weight: 40 },
            { category: checkMoves, weight: 30 },
            { category: castlingMoves, weight: 20 },
            { category: centerMoves, weight: 15 },
            { category: developingMoves, weight: 10 },
            { category: otherMoves, weight: 50 }
          ];
        }
        
        // Weight-based move selection with adjusted methodology
        selectedMove = improvedWeightedMoveSelection(moveProbabilities, difficulty);
      }
      
      console.log("Fallback AI selected move:", selectedMove);
      
      if (selectedMove) {
        // Verify the move is legal before making it
        const testGame = new Chess(gameRef.current.fen());
        try {
          const legalMove = testGame.move({
            from: selectedMove.from,
            to: selectedMove.to,
            promotion: 'q'
          });
          
          if (legalMove) {
            console.log("Making valid move:", selectedMove.from, "to", selectedMove.to);
            makeMove({
              from: selectedMove.from,
              to: selectedMove.to,
              promotion: 'q'
            });
          } else {
            console.error("Move was rejected by chess.js:", selectedMove);
            // Pick another random legal move as fallback
            const legalMoves = gameRef.current.moves({ verbose: true });
            if (legalMoves.length > 0) {
              const randomMove = legalMoves[Math.floor(Math.random() * legalMoves.length)];
              makeMove({
                from: randomMove.from,
                to: randomMove.to,
                promotion: randomMove.promotion || 'q'
              });
            }
          }
        } catch (error) {
          console.error("Error checking move validity:", error);
          // Pick a random legal move as fallback
          const legalMoves = gameRef.current.moves({ verbose: true });
          if (legalMoves.length > 0) {
            const randomMove = legalMoves[Math.floor(Math.random() * legalMoves.length)];
            makeMove({
              from: randomMove.from,
              to: randomMove.to,
              promotion: randomMove.promotion || 'q'
            });
          }
        }
      } else {
        // Fallback to completely random move if all else fails
        const randomMove = legalMoves[Math.floor(Math.random() * legalMoves.length)];
        console.log("No move selected, making random move:", randomMove);
        makeMove({
          from: randomMove.from,
          to: randomMove.to,
          promotion: 'q'
        });
      }
      
      setIsThinking(false);
    }, 500 + Math.random() * 500); // Varied thinking time for realism
  };
  
  // Improved weighted move selection with blunder probability
  const improvedWeightedMoveSelection = (moveProbabilities, elo) => {
    // Calculate total weight
    let totalWeight = 0;
    const validCategories = moveProbabilities.filter(item => item.category.length > 0);
    
    validCategories.forEach(item => {
      totalWeight += item.weight;
    });
    
    if (totalWeight === 0 || validCategories.length === 0) {
      return null;
    }
    
    // Blunder probability - higher at lower ELO
    const blunderChance = elo <= 800 ? 0.3 : 
                          elo <= 1200 ? 0.15 : 
                          elo <= 1500 ? 0.08 : 
                          elo <= 2000 ? 0.03 :
                          0.01; // 2200+ should rarely blunder
    
    // Deliberately make a poor move sometimes based on ELO
    if (Math.random() < blunderChance) {
      // Find the worst category available (usually "otherMoves")
      const worstCategory = validCategories.find(item => 
        item.category === otherMoves || item.category.some(move => move.score < 0)
      ) || validCategories[validCategories.length - 1];
      
      if (worstCategory && worstCategory.category.length > 0) {
        // Pick random move from worst category
        return worstCategory.category[Math.floor(Math.random() * worstCategory.category.length)].move;
      }
    }
    
    // Choose a category based on weights
    const randomNum = Math.random() * totalWeight;
    let weightSum = 0;
    let chosenCategory = null;
    
    for (const item of validCategories) {
      weightSum += item.weight;
      if (randomNum <= weightSum) {
        chosenCategory = item.category;
        break;
      }
    }
    
    if (!chosenCategory || chosenCategory.length === 0) {
      return null;
    }
    
    // Sort moves in the chosen category by score in descending order
    chosenCategory.sort((a, b) => b.score - a.score);
    
    // Choose one of the top moves in the category, with higher probability for better moves
    // Adjust selection based on ELO
    let moveQuality;
    if (elo >= 2200) {
      // Expert - almost always picks best move
      moveQuality = Math.pow(Math.random(), 5); // Heavily skewed toward 0 (best moves)
    } else if (elo >= 1500) {
      // Advanced - usually picks good moves
      moveQuality = Math.pow(Math.random(), 3); // Skewed toward best moves
    } else if (elo >= 1200) {
      // Intermediate - mix of good and mediocre moves
      moveQuality = Math.pow(Math.random(), 1.5); // Slightly skewed toward best moves
    } else {
      // Beginner - often picks mediocre moves
      moveQuality = Math.random(); // Linear distribution, no skew
    }
    
    const topMovesCount = Math.min(chosenCategory.length, elo >= 1500 ? 2 : 3);
    const randomIndex = Math.floor(moveQuality * topMovesCount);
    
    return chosenCategory[randomIndex].move;
  };
  
  // Helper function to get material count
  const getMaterialCount = (chessGame) => {
    const pieceValues = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };
    let whiteValue = 0;
    let blackValue = 0;
    
    const board = chessGame.board();
    
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const square = board[row][col];
        if (square) {
          const value = pieceValues[square.type] || 0;
          if (square.color === 'w') {
            whiteValue += value;
          } else {
            blackValue += value;
          }
        }
      }
    }
    
    return { white: whiteValue, black: blackValue };
  };
  
  // Helper function to get piece value
  const getPieceValue = (pieceType) => {
    const pieceValues = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };
    return pieceValues[pieceType] || 0;
  };
  
  // Helper function to count pieces still in home row
  const countPiecesInHomeRow = (chessGame) => {
    const board = chessGame.board();
    let count = 0;
    
    // Check white's home row (row 7 in zero-indexed board array)
    if (chessGame.turn() === 'w') {
      for (let col = 0; col < 8; col++) {
        const piece = board[7][col];
        if (piece && piece.color === 'w' && piece.type !== 'p') {
          count++;
        }
      }
    } 
    // Check black's home row (row 0 in zero-indexed board array)
    else {
      for (let col = 0; col < 8; col++) {
        const piece = board[0][col];
        if (piece && piece.color === 'b' && piece.type !== 'p') {
          count++;
        }
      }
    }
    
    return count;
  };

  // Helper function to check if a square is under attack
  const isSquareAttacked = (chessGame, square) => {
    const piece = chessGame.get(square);
    if (!piece) return false;
    
    const color = piece.color;
    
    // Get opponent color
    const opponentColor = color === 'w' ? 'b' : 'w';
    
    // Check if any opponent piece can capture this square
    const attackingMoves = [];
    const board = chessGame.board();
    
    // Check all squares on the board
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const piece = board[row][col];
        // If there's a piece and it's the opponent's
        if (piece && piece.color === opponentColor) {
          const from = String.fromCharCode(97 + col) + (8 - row); // Convert to algebraic notation
          
          try {
            // Get all legal moves from this square
            const moves = chessGame.moves({
              square: from,
              verbose: true
            });
            
            // Check if any move can capture at our target square
            for (const move of moves) {
              if (move.to === square) {
                attackingMoves.push(move);
              }
            }
          } catch (error) {
            // Ignore errors in attack detection
          }
        }
      }
    }
    
    return attackingMoves.length > 0;
  };

  // Handle piece selection to show possible moves
  const onSquareClick = (square) => {
    if (analysisMode || !gameStarted || isThinking) return;
    
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
    if (analysisMode || isThinking || !gameStarted) return false;
    
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

  // Start a new game
  const startNewGame = () => {
    gameRef.current = new Chess();
    setGame(new Chess());
    setFen(gameRef.current.fen());
    setGameStatus('Game started');
    setMoveHistory([]);
    setBestMoves([]);
    setEvaluation(0);
    setPossibleMoves([]);
    setSelectedSquare(null);
    setFullMoveHistory([]);
    setCurrentMoveIndex(0);
    setAnalysisMode(false);
    setGameStarted(true);
    
    // If player is black, get the bot's first move
    if (playerColor === 'black') {
      setTimeout(() => getBotMove(), 300);
    }
  };

  // Reset the game
  const resetGame = () => {
    setGameStarted(false);
    setAnalysisMode(false);
    gameRef.current = new Chess();
    setGame(new Chess());
    setFen(gameRef.current.fen());
    setGameStatus('Choose your color and difficulty to start');
    setMoveHistory([]);
    setBestMoves([]);
    setEvaluation(0);
    setPossibleMoves([]);
    setSelectedSquare(null);
    setFullMoveHistory([]);
    setCurrentMoveIndex(0);
  };

  // Toggle analysis mode
  const toggleAnalysisMode = () => {
    const newMode = !analysisMode;
    setAnalysisMode(newMode);
    
    if (newMode) {
      // Save current position
      setCurrentMoveIndex(fullMoveHistory.length);
    } else {
      // Return to current game
      gameRef.current = new Chess();
      for (const move of fullMoveHistory) {
        gameRef.current.move(move);
      }
      setGame(new Chess(gameRef.current.fen()));
      updateGameState();
    }
  };

  // Navigate through move history in analysis mode
  const goToMove = (index) => {
    if (!analysisMode || !fullMoveHistory) return;
    
    const tempGame = new Chess();
    
    // Apply moves up to the selected index
    for (let i = 0; i <= index && i < fullMoveHistory.length; i++) {
      tempGame.move(fullMoveHistory[i]);
    }
    
    gameRef.current = tempGame;
    setGame(new Chess(tempGame.fen()));
    setFen(tempGame.fen());
    setCurrentMoveIndex(index);
    
    // Analyze the position
    if (engineRef.current && engineLoaded) {
      setBestMoves([]);
      setEvaluation(0);
      
      // Reset engine for clean analysis
      engineRef.current.postMessage('ucinewgame');
      
      const moves = tempGame.history();
      engineRef.current.postMessage('position fen ' + tempGame.fen());
      engineRef.current.postMessage('go depth 15');
    }
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
    // For empty squares
    if (!gameRef.current.get(square)) {
      customSquareStyles[square] = {
        background: 'radial-gradient(circle at center, rgba(0,0,0,0.2) 20%, transparent 30%)',
      };
    } else {
      // For captures
      customSquareStyles[square] = {
        background: 'radial-gradient(transparent 0%, transparent 79%, rgba(220,0,0,0.3) 80%)',
      };
    }
  });

  // Last move highlighting
  if (fullMoveHistory.length > 0 && !analysisMode) {
    const lastMove = fullMoveHistory[fullMoveHistory.length - 1];
    if (lastMove) {
      customSquareStyles[lastMove.from] = {
        backgroundColor: 'rgba(255, 170, 0, 0.2)',
      };
      customSquareStyles[lastMove.to] = {
        backgroundColor: 'rgba(255, 170, 0, 0.4)',
      };
    }
  } else if (analysisMode && currentMoveIndex > 0 && currentMoveIndex <= fullMoveHistory.length) {
    const move = fullMoveHistory[currentMoveIndex - 1];
    if (move) {
      customSquareStyles[move.from] = {
        backgroundColor: 'rgba(255, 170, 0, 0.2)',
      };
      customSquareStyles[move.to] = {
        backgroundColor: 'rgba(255, 170, 0, 0.4)',
      };
    }
  }

  return (
    <div className="chess-game-container">
      <div className="game-controls">
        <div className="control-panel">
          <h2>Play Chess</h2>
          {engineError && <p className="engine-error">{engineError}</p>}
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
                  <div className="button-group difficulty-buttons">
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
                    <button 
                      className={difficulty === 2200 ? 'active' : ''}
                      onClick={() => setDifficulty(2200)}
                    >
                      Expert (2200+)
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
              
              {bestMoves.length > 0 && (
                <div className="best-moves">
                  <h4>Top Engine Moves:</h4>
                  <ul>
                    {bestMoves.map((moveInfo, idx) => (
                      <li key={idx}>
                        <span className="move-notation">{moveInfo.san || moveInfo.uci}</span>
                        <span className="move-eval">{formatEvaluation(moveInfo.score, moveInfo.isMate)}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              
              <div className="game-buttons">
                <button className="secondary-button" onClick={resetGame}>
                  Reset Game
                </button>
                <button 
                  className={`secondary-button ${analysisMode ? 'analysis-active' : ''}`}
                  onClick={toggleAnalysisMode}
                >
                  {analysisMode ? 'Back to Game' : 'Analysis Mode'}
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
                  {moveHistory.map((move, index) => (
                    <tr 
                      key={move.moveNumber}
                      className={analysisMode && Math.floor(currentMoveIndex / 2) === index ? 'highlighted-row' : ''}
                    >
                      <td>{move.moveNumber}.</td>
                      <td 
                        className={analysisMode && currentMoveIndex === index * 2 ? 'highlighted-move' : ''}
                        onClick={() => analysisMode && move.whiteMove && goToMove(index * 2)}
                      >
                        {move.whiteMove}
                      </td>
                      <td 
                        className={analysisMode && currentMoveIndex === index * 2 + 1 ? 'highlighted-move' : ''}
                        onClick={() => analysisMode && move.blackMove && goToMove(index * 2 + 1)}
                      >
                        {move.blackMove}
                      </td>
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
          animationDuration={200}
          boardWidth={560}
        />
        
        {analysisMode && (
          <div className="analysis-controls">
            <button onClick={() => goToMove(0)} disabled={currentMoveIndex === 0}>
              &lt;&lt;
            </button>
            <button onClick={() => goToMove(currentMoveIndex - 1)} disabled={currentMoveIndex === 0}>
              &lt;
            </button>
            <span className="move-counter">
              Move {Math.floor(currentMoveIndex / 2) + 1}
              {currentMoveIndex % 2 === 0 ? ' (White)' : ' (Black)'}
            </span>
            <button onClick={() => goToMove(currentMoveIndex + 1)} disabled={currentMoveIndex >= fullMoveHistory.length}>
              &gt;
            </button>
            <button onClick={() => goToMove(fullMoveHistory.length)} disabled={currentMoveIndex >= fullMoveHistory.length}>
              &gt;&gt;
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default Board;
export { Board };