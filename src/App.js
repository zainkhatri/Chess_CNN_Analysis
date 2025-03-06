import React from 'react';
import Board from './components/Board'; // adjust path if needed
import './App.css';

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>Chess Game</h1>
      </header>
      <main className="main-content">
        <Board />
      </main>
      <footer className="app-footer">
        <p>Chess Post-Game Analysis App</p>
      </footer>
    </div>
  );
}

export default App;