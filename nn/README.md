# Training Progress Report

## Overview

This document tracks multiple training runs of our chess CNN model. The model was trained on a dataset of **3,232 chess positions** and evaluated using a **policy head (move prediction) and a value head (position evaluation).** Each run experimented with different hyperparameters to optimize performance.

---

## Run #1

The first training run was executed with the following hyperparameters:

- **Batch Size:** 16  
- **Number of Filters:** 64  
- **Number of Residual Blocks:** 6  
- **Epochs:** 20  
- **Device Used:** Apple Silicon GPU (MPS)  

### Dataset Details

- **Total Positions:** 3,232  
- **Training Samples:** 2,585  
- **Validation Samples:** 647  
- **Tensor Shapes:**  
  - Positions: `(3232, 12, 8, 8)`
  - Evaluations: `(3232)`
  - Moves: `(3232)`

### Training Process

The model training started with an initial validation loss of **0.8853** and a **policy accuracy of 0.0244**. Early epochs showed improvement, but later epochs failed to reduce validation loss. The training stopped early at **7 epochs** due to a lack of validation loss improvement.

#### Key Metrics Across Epochs

| Epoch | Train Loss | Val Loss | Policy Accuracy | Value MAE |
|-------|-----------|---------|----------------|------------|
| 1     | 0.8616    | 0.8853  | 0.0208         | 0.0014     |
| 2     | 0.6404    | 0.8812  | 0.0343         | 0.0016     |
| 3     | 0.4797    | 0.8817  | 0.0625         | 0.0012     |
| 4     | 0.2645    | 0.9886  | 0.1244         | 0.0009     |
| 5     | 0.1423    | 1.0520  | 0.1778         | 0.0006     |
| 6     | 0.0772    | 0.9781  | 0.2043         | 0.0006     |
| 7     | 0.0299    | 1.0430  | 0.2246         | 0.0005     |

#### Observations:

- **Training Loss consistently decreased**, showing the model was learning.
- **Validation Loss did not improve after epoch 3**, indicating overfitting.
- **Policy Accuracy improved** but remained relatively low.
- **Value MAE gradually decreased, improving evaluation accuracy.**
- **Training took ~2 minutes and 24 seconds.**

---

## Run #2

The second training run adjusted hyperparameters for better generalization:

- **Batch Size:** 16  
- **Number of Filters:** 48  
- **Number of Residual Blocks:** 4  
- **Epochs:** 30  
- **Learning Rate:** 0.0005  
- **Device Used:** Apple Silicon GPU (MPS)  

### Training Process

This run started with a **higher initial validation loss (0.9758)** than the previous run. The model showed faster training times and better policy accuracy, but validation loss fluctuated. Training stopped early at **10 epochs** due to lack of improvement.

#### Key Metrics Across Epochs

| Epoch | Train Loss | Val Loss | Policy Accuracy | Value MAE |
|-------|-----------|---------|----------------|------------|
| 1     | 0.8521    | 0.9277  | 0.0243         | 0.0030     |
| 2     | 0.4387    | 0.8841  | 0.0763         | 0.0021     |
| 3     | 0.1663    | 0.9108  | 0.1755         | 0.0017     |
| 4     | 0.0532    | 0.9128  | 0.2151         | 0.0013     |
| 5     | 0.0232    | 0.8489  | 0.2270         | 0.0010     |
| 6     | 0.0177    | 0.8953  | 0.2272         | 0.0009     |
| 7     | 0.0139    | 0.9544  | 0.2302         | 0.0008     |
| 8     | 0.0041    | 0.9764  | 0.2317         | 0.0008     |
| 9     | 0.0030    | 0.9134  | 0.2320         | 0.0008     |
| 10    | 0.0023    | 0.9394  | 0.2311         | 0.0006     |

#### Observations:

- **Faster training speed** (~37 seconds total).
- **Better policy accuracy (23.2%)** than the first run.
- **Validation loss did not consistently decrease**, leading to early stopping.
- **Value MAE improved significantly, indicating more precise evaluations.**

---

## Summary & Next Steps

### Improvements Between Runs:
- **Training time reduced from 2m 24s to 37s.**
- **Policy accuracy improved (~23.2% in Run #2 vs. ~22.4% in Run #1).**
- **Value MAE consistently decreased across epochs.**

### Remaining Issues:
- **Validation loss fluctuates, indicating overfitting.**
- **Despite higher policy accuracy, the validation performance did not generalize well.**

### Future Optimizations:
1. **Increase Dataset Size** – More training data might help generalization.
2. **Hyperparameter Tuning** – Adjusting learning rate, batch size, and filters further.
3. **Regularization Techniques** – Dropout, weight decay, or data augmentation.
4. **Alternative Architectures** – Testing deeper or different CNN structures.
5. **Experiment with Learning Rate Scheduling** – Dynamically adjust learning rate to optimize validation loss.

