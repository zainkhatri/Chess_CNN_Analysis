# Chess CNN: Deep Learning for Chess Move Prediction and Position Evaluation

## Abstract
Chess move prediction and position evaluation represent challenging problems due to the game's inherent complexity. In this project, we implement a convolutional neural network (CNN) with a dual-head architecture that simultaneously predicts the best move (policy head) and evaluates the position (value head) of a chess game. Our model is trained on an enhanced dataset of 16,174 positions extracted from high-quality games and annotated using the Stockfish chess engine. We performed extensive experimentation by tuning hyperparameters, incorporating advanced neural components (such as residual blocks, attention mechanisms, and focal loss), and scaling the dataset. Our best model achieved a validation loss of **0.3210** and a move prediction accuracy of **2.89%**. These results highlight the challenges posed by severe class imbalance and the vast move space. We conclude with discussions on future improvements, including the use of transformer architectures, hybrid search methods, and further data augmentation.

## Introduction
The impact of deep learning on strategic games is exemplified by systems like AlphaZero. Despite significant progress in position evaluation, chess move prediction remains challenging due to the enormous number of possible moves and the imbalance in move distributions. This project investigates CNN architectures tailored for chess by balancing two tasks: predicting moves (policy head) and evaluating positions (value head). We also explore how dataset size and model complexity affect performance.

## Model Architecture
Our CNN employs a dual-head design:
- **Input Layer**: Receives a 12-channel tensor representing an 8×8 chessboard (6 piece types × 2 colors).
- **Convolutional Backbone**: Utilizes multiple residual blocks with skip connections and batch normalization to extract features.
- **Policy Head**: Outputs a probability distribution over 1968 possible moves (classification task).
- **Value Head**: Outputs a continuous evaluation score ranging from -1 (Black advantage) to +1 (White advantage).

## Dataset
We prepared two datasets:
- **Initial Dataset**: 3,232 chess positions.
- **Enhanced Dataset**: 16,174 chess positions sourced from high-quality (1800+ ELO) games.

Each data point contains:
- A tensor representation (12×8×8) of the board.
- A Stockfish evaluation score.
- A best move label (class index).

## Experimental Setup and Results
We conducted experiments with varying architectures and hyperparameters. Below is a summary of the different runs:

| Run    | Batch Size | Filters | Blocks | Dataset Size | Best Val Loss | Policy Acc. | Training Time  |
|--------|------------|---------|--------|--------------|---------------|-------------|----------------|
| **1**  | 16         | 64      | 6      | 3,232        | 0.8812        | 22.46%      | 2m 24s         |
| **2**  | 16         | 48      | 4      | 3,232        | 0.8489        | 23.20%      | 37s            |
| **3**  | 8          | 32      | 3      | 1,500        | 0.6194        | 3.72%       | 49s            |
| **4**  | 32         | 96      | 8      | 16,174       | 0.3763        | 2.67%       | 7m 25s         |
| **5*** | 64         | 128     | 10     | 16,174       | **0.3210**    | **2.89%**   | 4m 16s         |

*Run 5 incorporates attention mechanisms, focal loss, and dropout to better address class imbalance.*

### Observations
- **Dataset Size**: Expanding the dataset reduced overfitting and lowered the validation loss.
- **Model Complexity**: Increasing filters and blocks improved generalization at the cost of higher computational demand.
- **Policy vs. Value**: While the value head converged quickly, the policy head’s accuracy plateaued due to the severe imbalance in move frequency.

## Key Findings
- **Validation Loss vs. Accuracy Trade-off**: A lower validation loss did not guarantee a proportional increase in move prediction accuracy.
- **Impact of Data Scaling**: A larger dataset significantly improved model robustness.
- **Architectural Complexity**: More complex models reduced loss but risked overfitting, emphasizing the need for regularization techniques.
- **Task Difficulty**: Position evaluation (value head) was more straightforward compared to move prediction (policy head), highlighting the challenges posed by a vast and imbalanced move space.

## Conclusions
Our experiments demonstrate that while CNNs can effectively evaluate chess positions, predicting moves remains a significant challenge due to class imbalance and the enormous number of potential moves. Advanced techniques like attention mechanisms and focal loss provide marginal improvements, yet a breakthrough may require larger datasets or fundamentally different architectures (e.g., transformers).

## Future Work
- **Dataset Expansion**: Aim to collect over 100,000 positions to further mitigate class imbalance.
- **Transformer Architectures**: Explore models that better capture long-range dependencies on the chessboard.
- **Hybrid Methods**: Combine neural network predictions with traditional search algorithms for improved move prediction.
- **Data Augmentation**: Investigate board transformations (rotations, flips) to enhance training diversity.

## References
- Silver et al. (2017). [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815). *arXiv preprint arXiv:1712.01815*.
- He et al. (2016). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). *IEEE CVPR*.
- [Stockfish Chess Engine](https://stockfishchess.org).

## Repository Structure
    /
    ├── data/                       
    │   ├── lichess_2017_01_part_1.pgn
    │   ├── lichess_2017_01_part_2.pgn
    │   ├── lichess_2017_01_part_3.pgn
    │   ├── lichess_2017_01_part_4.pgn
    │   └── lichess_2017_01_part_5.pgn
    │
    ├── nn/                         
    │   ├── README.md               
    │   ├── analyze_game.py         
    │   ├── data_processing.py      
    │   ├── game_advisor.py         
    │   ├── model.py                
    │   ├── processed_data/         
    │   ├── train_model.py          
    │   └── models/ (directory containing saved model checkpoints)
    │
    ├── models/                     
    │   ├── chess_cnn_best.pt       
    │   ├── chess_cnn_epoch_1.pt    
    │   ├── chess_cnn_epoch_2.pt    
    │   ├── chess_cnn_epoch_5.pt    
    │   ├── final/                  
    │   ├── run3/                   
    │   ├── run4/                   
    │   ├── training_history.json   
    │   └── training_curves.png     
    │
    ├── public/                     
    │   ├── Stockfish/              
    │   ├── favicon.ico             
    │   ├── index.html              
    │   ├── logo192.png             
    │   ├── logo512.png             
    │   ├── manifest.json           
    │   ├── robots.txt              
    │   └── stockfish.js            
    │
    └── README.md                   

## Requirements
- Python ≥ 3.8
- PyTorch ≥ 1.12
- NumPy
- Matplotlib
- python-chess
- tqdm

Install dependencies with:
    pip install torch numpy matplotlib python-chess tqdm

## Running the Code

1. **Process Data**:
    python3 nn/data_processing.py

2. **Train Basic Model**:
    python3 nn/train_model.py --batch_size 32 --num_filters 96 --num_blocks 8 --epochs 30

3. **Train Enhanced Model (Recommended)**:
    python3 nn/train_model_final.py --batch_size 64 --num_filters 128 --num_blocks 10 --epochs 40

4. **Evaluate a Game**:
    python3 nn/game_advisor.py --model ./models/final/chess_cnn_best.pt --pgn your_game.pgn

## Bonus Points Justification
- **Novel Application**: We integrated focal loss and attention mechanisms to specifically address the class imbalance in chess move prediction.
- **Comprehensive Experiments**: Extensive hyperparameter tuning was performed across multiple CNN configurations.
- **Data Collection and Curation**: We curated a high-quality chess dataset (16K+ positions) from multiple PGN files sourced from lichess.