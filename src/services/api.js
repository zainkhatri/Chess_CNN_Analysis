import axios from 'axios';

// Create axios instance with base URL
const apiClient = axios.create({
  baseURL: process.env.REACT_APP_API_URL || '/api',
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
  }
});

// API service methods
const api = {
  // Health check
  checkHealth: () => {
    return apiClient.get('/health');
  },
  
  // Analyze a position
  analyzePosition: (fen) => {
    return apiClient.post('/analyze/position', { fen });
  },
  
  // Analyze a game
  analyzeGame: (pgn) => {
    return apiClient.post('/analyze/game', { pgn });
  },
  
  // Get a saved analysis
  getAnalysis: (id) => {
    return apiClient.get(`/analysis/${id}`);
  },
  
  // Get a move from the bot
  getBotMove: (fen, difficulty) => {
    return apiClient.post('/bot/move', { fen, difficulty });
  },
  
  // Save a completed game
  saveGame: (pgn) => {
    return apiClient.post('/save/game', { pgn });
  }
};

export default api;