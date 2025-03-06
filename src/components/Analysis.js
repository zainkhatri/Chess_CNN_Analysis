import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Chess } from 'chess.js';
import { Chessboard } from 'react-chessboard';
import Chart from 'chart.js/auto';
import api from '../services/api';
import Heatmap from './Heatmap';
import MoveQualityChart from './MoveQualityChart';
import './Analysis.css';
import './AccuracyGraph.css';

// AccuracyGraph Component
function AccuracyGraph({ positions, showFullWidth = false, onSelectPosition }) {
  const chartRef = useRef(null);
  const chartInstance = useRef(null);

  useEffect(() => {
    if (!positions || positions.length === 0) {
      return;
    }

    // Extract evaluation data
    const labels = positions.map((p, index) => 
      `${Math.floor(index / 2) + 1}${index % 2 === 0 ? '.' : '...'}`
    );
    
    const evaluations = positions.map(p => 
      p.analysis?.stockfish_eval || 0
    );

    // Determine quality colors
    const pointBackgroundColors = positions.map(p => {
      const quality = p.evaluation?.quality;
      if (!quality) return 'rgba(54, 162, 235, 0.7)'; // Default blue
      
      switch (quality) {
        case 'best':
          return 'rgba(75, 192, 192, 0.7)'; // Green
        case 'inaccuracy':
          return 'rgba(255, 206, 86, 0.7)'; // Yellow
        case 'mistake':
          return 'rgba(255, 159, 64, 0.7)'; // Orange
        case 'blunder':
          return 'rgba(255, 99, 132, 0.7)'; // Red
        default:
          return 'rgba(54, 162, 235, 0.7)'; // Blue
      }
    });

    // Create the chart
    if (chartRef.current) {
      // Destroy previous chart if it exists
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }

      const ctx = chartRef.current.getContext('2d');
      chartInstance.current = new Chart(ctx, {
        type: 'line',
        data: {
          labels: labels,
          datasets: [
            {
              label: 'Position Evaluation',
              data: evaluations,
              borderColor: 'rgba(54, 162, 235, 1)',
              backgroundColor: pointBackgroundColors,
              borderWidth: 2,
              pointRadius: 5,
              pointHoverRadius: 7,
              tension: 0.1,
              fill: false
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          aspectRatio: showFullWidth ? 2.5 : 1.5,
          scales: {
            y: {
              title: {
                display: true,
                text: 'Evaluation (pawns)'
              },
              suggestedMin: -5,
              suggestedMax: 5
            }
          },
          plugins: {
            legend: {
              display: false
            },
            tooltip: {
              callbacks: {
                title: function(tooltipItems) {
                  const idx = tooltipItems[0].dataIndex;
                  const moveNumber = Math.floor(idx / 2) + 1;
                  const side = idx % 2 === 0 ? 'White' : 'Black';
                  return `Move ${moveNumber} (${side})`;
                },
                label: function(context) {
                  const idx = context.dataIndex;
                  const evalValue = context.parsed.y;
                  const formattedEval = evalValue > 0 ? `+${evalValue.toFixed(1)}` : evalValue.toFixed(1);
                  
                  if (positions[idx]?.evaluation?.quality) {
                    const quality = positions[idx].evaluation.quality;
                    return [
                      `Evaluation: ${formattedEval}`,
                      `Quality: ${quality.charAt(0).toUpperCase() + quality.slice(1)}`
                    ];
                  }
                  return `Evaluation: ${formattedEval}`;
                }
              }
            }
          },
          onClick: (event, elements) => {
            if (elements && elements.length > 0 && onSelectPosition) {
              const index = elements[0].index;
              onSelectPosition(index);
            }
          }
        }
      });
    }

    // Clean up on unmount
    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, [positions, showFullWidth, onSelectPosition]);

  return (
    <div className={`accuracy-chart-wrapper ${showFullWidth ? 'full-width' : ''}`}>
      <canvas ref={chartRef} />
      <div className="chart-legend">
        <div className="legend-item">
          <span className="color-dot best-move"></span>
          <span>Best Move</span>
        </div>
        <div className="legend-item">
          <span className="color-dot inaccuracy"></span>
          <span>Inaccuracy</span>
        </div>
        <div className="legend-item">
          <span className="color-dot mistake"></span>
          <span>Mistake</span>
        </div>
        <div className="legend-item">
          <span className="color-dot blunder"></span>
          <span>Blunder</span>
        </div>
      </div>
    </div>
  );
}

// Analysis Component
function Analysis() {
  const { id } = useParams();
  const navigate = useNavigate();
  
  const [isLoading, setIsLoading] = useState(true);
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState(null);
  const [currentMoveIndex, setCurrentMoveIndex] = useState(0);
  const [game, setGame] = useState(new Chess());
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');
  
  // Load analysis data
  useEffect(() => {
    const loadAnalysis = async () => {
      if (!id) {
        setError("No analysis ID provided. Please upload a game for analysis.");
        setIsLoading(false);
        return;
      }
      
      try {
        const response = await api.getAnalysis(id);
        setAnalysis(response.data);
        
        // Initialize the chess board with the starting position
        if (response.data.positions && response.data.positions.length > 0) {
          const firstPosition = response.data.positions[0];
          const newGame = new Chess();
          newGame.load(firstPosition.fen);
          setGame(newGame);
        }
        
        setIsLoading(false);
      } catch (error) {
        console.error('Error loading analysis:', error);
        setError("Failed to load the analysis. Please try again.");
        setIsLoading(false);
      }
    };
    
    loadAnalysis();
  }, [id]);
  
  // Handle move navigation
  const goToMove = (index) => {
    if (!analysis || !analysis.positions || index < 0 || index >= analysis.positions.length) {
      return;
    }
    
    const position = analysis.positions[index];
    const newGame = new Chess();
    newGame.load(position.fen);
    setGame(newGame);
    setCurrentMoveIndex(index);
    
    // Toggle heatmap off when navigating
    setShowHeatmap(false);
  };
  
  const goToPreviousMove = () => {
    goToMove(currentMoveIndex - 1);
  };
  
  const goToNextMove = () => {
    goToMove(currentMoveIndex + 1);
  };
  
  const goToFirstMove = () => {
    goToMove(0);
  };
  
  const goToLastMove = () => {
    if (analysis && analysis.positions) {
      goToMove(analysis.positions.length - 1);
    }
  };
  
  // Toggle the heatmap display
  const toggleHeatmap = () => {
    setShowHeatmap(!showHeatmap);
  };
  
  // Calculate the evaluation bar value (from -1 to 1)
  const calculateEvalBarValue = (evalScore) => {
    // Convert from raw eval (in pawns) to a value between -1 and 1 for the UI
    return Math.tanh(evalScore / 3);
  };
  
  // Format the evaluation display
  const formatEvaluation = (evalScore) => {
    if (Math.abs(evalScore) > 5) {
      return evalScore > 0 ? `+M${Math.ceil((100 - Math.abs(evalScore)) / 2)}` : `-M${Math.ceil((100 - Math.abs(evalScore)) / 2)}`;
    }
    return evalScore > 0 ? `+${evalScore.toFixed(1)}` : evalScore.toFixed(1);
  };
  
  // Get current position data
  const getCurrentPosition = () => {
    if (!analysis || !analysis.positions || currentMoveIndex >= analysis.positions.length) {
      return null;
    }
    return analysis.positions[currentMoveIndex];
  };
  
  const currentPosition = getCurrentPosition();
  
  if (isLoading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Loading analysis...</p>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="error-container">
        <h2>Error</h2>
        <p>{error}</p>
        <button onClick={() => navigate('/play')}>Start a new game</button>
      </div>
    );
  }
  
  if (!analysis) {
    return (
      <div className="analysis-empty">
        <h2>No Analysis Available</h2>
        <p>No analysis data is available. Please play a game first.</p>
        <button onClick={() => navigate('/play')}>Start a new game</button>
      </div>
    );
  }
  
  return (
    <div className="game-analysis-container">
      <div className="analysis-header">
        <h2>Game Analysis</h2>
        <div className="game-metadata">
          <p><strong>White:</strong> {analysis.metadata?.white || 'Unknown'}</p>
          <p><strong>Black:</strong> {analysis.metadata?.black || 'Unknown'}</p>
          <p><strong>Date:</strong> {analysis.metadata?.date || 'Unknown'}</p>
          <p><strong>Result:</strong> {analysis.metadata?.result || 'Unknown'}</p>
        </div>
      </div>
      
      <div className="analysis-tabs">
        <button 
          className={`tab-button ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </button>
        <button 
          className={`tab-button ${activeTab === 'moveByMove' ? 'active' : ''}`}
          onClick={() => setActiveTab('moveByMove')}
        >
          Move-by-Move Analysis
        </button>
        <button 
          className={`tab-button ${activeTab === 'charts' ? 'active' : ''}`}
          onClick={() => setActiveTab('charts')}
        >
          Performance Charts
        </button>
      </div>
      
      <div className="analysis-content">
        {activeTab === 'overview' && (
          <div className="overview-tab">
            <div className="summary-section">
              <h3>Game Summary</h3>
              <div className="accuracy-section">
                <div className="player-accuracy">
                  <h4>White</h4>
                  <div className="accuracy-value">{analysis.summary?.white_accuracy?.toFixed(1)}%</div>
                  <div className="mistake-summary">
                    <p>{analysis.summary?.white_mistakes || 0} mistakes</p>
                    <p>{analysis.summary?.white_blunders || 0} blunders</p>
                  </div>
                </div>
                <div className="player-accuracy">
                  <h4>Black</h4>
                  <div className="accuracy-value">{analysis.summary?.black_accuracy?.toFixed(1)}%</div>
                  <div className="mistake-summary">
                    <p>{analysis.summary?.black_mistakes || 0} mistakes</p>
                    <p>{analysis.summary?.black_blunders || 0} blunders</p>
                  </div>
                </div>
              </div>
              
              <div className="decisive-moments">
                <h4>Key Moments</h4>
                {analysis.summary?.decisive_moments?.length > 0 ? (
                  <ul className="moments-list">
                    {analysis.summary.decisive_moments.map((moment, index) => (
                      <li key={index} className="moment-item">
                        <span className="move-number">Move {Math.floor(moment.move_number / 2) + 1} ({moment.move_number % 2 === 1 ? 'White' : 'Black'})</span>
                        <span className={`quality-label ${moment.quality}`}>{moment.quality}</span>
                        <span className="loss-value">Lost {moment.centipawn_loss.toFixed(1)} pawns</span>
                        <button 
                          className="view-position-button"
                          onClick={() => {
                            // Find the position index corresponding to this moment
                            const posIndex = analysis.positions.findIndex(
                              p => p.move_number === moment.move_number
                            );
                            if (posIndex >= 0) {
                              setActiveTab('moveByMove');
                              goToMove(posIndex);
                            }
                          }}
                        >
                          View Position
                        </button>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p>No decisive moments found in this game.</p>
                )}
              </div>
            </div>
            
            <div className="evaluation-chart">
              <h3>Game Evaluation</h3>
              <AccuracyGraph 
                positions={analysis.positions} 
                onSelectPosition={(index) => {
                  setActiveTab('moveByMove');
                  goToMove(index);
                }}
              />
            </div>
          </div>
        )}
        
        {activeTab === 'moveByMove' && (
          <div className="move-by-move-tab">
            <div className="board-analysis-container">
              <div className="board-with-eval">
                <div className="evaluation-bar">
                  <div 
                    className="eval-bar-fill" 
                    style={{ 
                      height: `${(1 - calculateEvalBarValue(currentPosition?.analysis?.stockfish_eval || 0)) * 50}%`
                    }}
                  ></div>
                  <div className="eval-value">
                    {formatEvaluation(currentPosition?.analysis?.stockfish_eval || 0)}
                  </div>
                </div>
                
                <div className="chessboard-wrapper">
                  <Chessboard
                    id="analysis-board"
                    position={game.fen()}
                    boardWidth={480}
                    arePiecesDraggable={false}
                  />
                  
                  {showHeatmap && currentPosition?.analysis?.heatmap && (
                    <div className="heatmap-overlay">
                      <Heatmap 
                        data={currentPosition.analysis.heatmap.heatmap} 
                        squareSize={60}
                      />
                    </div>
                  )}
                </div>
              </div>
              
              <div className="move-controls">
                <button onClick={goToFirstMove} disabled={currentMoveIndex === 0}>
                  &lt;&lt;
                </button>
                <button onClick={goToPreviousMove} disabled={currentMoveIndex === 0}>
                  &lt;
                </button>
                <span className="move-counter">
                  Move {Math.floor(currentMoveIndex / 2) + 1}
                  {currentMoveIndex % 2 === 0 ? ' (White)' : ' (Black)'}
                </span>
                <button onClick={goToNextMove} disabled={!analysis.positions || currentMoveIndex === analysis.positions.length - 1}>
                  &gt;
                </button>
                <button onClick={goToLastMove} disabled={!analysis.positions || currentMoveIndex === analysis.positions.length - 1}>
                  &gt;&gt;
                </button>
                
                <button 
                  className={`toggle-heatmap ${showHeatmap ? 'active' : ''}`}
                  onClick={toggleHeatmap}
                >
                  {showHeatmap ? 'Hide Heatmap' : 'Show Heatmap'}
                </button>
              </div>
            </div>
            
            <div className="position-analysis">
              {currentPosition ? (
                <>
                  <div className="move-quality">
                    {currentPosition.evaluation && (
                      <div className={`quality-indicator ${currentPosition.evaluation.quality}`}>
                        {currentPosition.evaluation.quality === 'best' ? 'Best Move' :
                          currentPosition.evaluation.quality === 'inaccuracy' ? 'Inaccuracy' :
                          currentPosition.evaluation.quality === 'mistake' ? 'Mistake' :
                          currentPosition.evaluation.quality === 'blunder' ? 'Blunder' : ''}
                          
                        {currentPosition.evaluation.quality !== 'best' && (
                          <span className="centipawn-loss">
                            -{currentPosition.evaluation.centipawn_loss.toFixed(1)}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                  
                  <div className="position-details">
                    <h4>Position Analysis</h4>
                    <div className="evaluation-comparison">
                      <div className="eval-item">
                        <span className="eval-label">Stockfish:</span>
                        <span className="eval-value">
                          {formatEvaluation(currentPosition.analysis?.stockfish_eval || 0)}
                        </span>
                      </div>
                      <div className="eval-item">
                        <span className="eval-label">Model:</span>
                        <span className="eval-value">
                          {formatEvaluation(currentPosition.analysis?.model_eval || 0)}
                        </span>
                      </div>
                    </div>
                    
                    <div className="material-balance">
                      <h5>Material Balance</h5>
                      <p>
                        White: {currentPosition.analysis?.material_balance?.white_material || 0} pawns
                        | Black: {currentPosition.analysis?.material_balance?.black_material || 0} pawns
                        | Difference: {(currentPosition.analysis?.material_balance?.balance || 0) > 0 ? '+' : ''}
                        {currentPosition.analysis?.material_balance?.balance || 0} pawns
                      </p>
                    </div>
                    
                    <div className="suggested-moves">
                      <h5>Best Moves</h5>
                      {currentPosition.analysis?.top_moves && currentPosition.analysis.top_moves.length > 0 ? (
                        <ul className="moves-list">
                          {currentPosition.analysis.top_moves.map((move, idx) => (
                            <li key={idx} className="move-item">
                              <span className="move-notation">{move.move}</span>
                              <span className="move-evaluation">{formatEvaluation(move.score / 100)}</span>
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p>No move suggestions available</p>
                      )}
                    </div>
                    
                    <div className="position-insights">
                      <h5>Position Insights</h5>
                      <ul className="insights-list">
                        {currentPosition.analysis?.mobility && (
                          <li>
                            Mobility: White has {currentPosition.analysis.mobility.white_total} legal moves,
                            Black has {currentPosition.analysis.mobility.black_total} legal moves
                          </li>
                        )}
                        
                        {currentPosition.analysis?.pawn_structure && (
                          <>
                            {currentPosition.analysis.pawn_structure.white_passed.length > 0 && (
                              <li>White has passed pawns on: {currentPosition.analysis.pawn_structure.white_passed.join(', ')}</li>
                            )}
                            {currentPosition.analysis.pawn_structure.black_passed.length > 0 && (
                              <li>Black has passed pawns on: {currentPosition.analysis.pawn_structure.black_passed.join(', ')}</li>
                            )}
                            {currentPosition.analysis.pawn_structure.white_isolated.length > 0 && (
                              <li>White has isolated pawns on: {currentPosition.analysis.pawn_structure.white_isolated.join(', ')}</li>
                            )}
                            {currentPosition.analysis.pawn_structure.black_isolated.length > 0 && (
                              <li>Black has isolated pawns on: {currentPosition.analysis.pawn_structure.black_isolated.join(', ')}</li>
                            )}
                          </>
                        )}
                      </ul>
                    </div>
                  </div>
                </>
              ) : (
                <p>Select a position to view analysis</p>
              )}
            </div>
          </div>
        )}
        
        {activeTab === 'charts' && (
          <div className="charts-tab">
            <div className="chart-container">
              <h3>Move Quality Distribution</h3>
              <MoveQualityChart 
                positions={analysis.positions}
              />
              <p className="chart-explanation">
                This chart shows the distribution of move quality throughout the game. 
                Best moves maintain advantage, while inaccuracies, mistakes, and blunders progressively lose advantage.
              </p>
            </div>
            
            <div className="chart-container">
              <h3>Evaluation Timeline</h3>
              <AccuracyGraph 
                positions={analysis.positions}
                showFullWidth={true}
                onSelectPosition={(index) => {
                  setActiveTab('moveByMove');
                  goToMove(index);
                }}
              />
              <p className="chart-explanation">
                This chart shows how the evaluation of the position changed throughout the game.
                Positive values favor White, negative values favor Black.
              </p>
            </div>
          </div>
        )}
      </div>
      
      <div className="analysis-actions">
        <button className="primary-button" onClick={() => navigate('/play')}>
          Play New Game
        </button>
      </div>
    </div>
  );
}

export { AccuracyGraph, Analysis };
export default Analysis;