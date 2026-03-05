# CS4375 Assignment — Neural Network Text Classification

**Author:** Oghenetejiri Etaghene
**NetID:** OXE220001
**Course:** CS4375 — Introduction to Machine Learning, Spring 2026

---

## Overview

This project implements two neural network architectures for 5-class sentiment analysis on Yelp restaurant reviews:

1. **Feed-Forward Neural Network (FFNN)** — Uses bag-of-words input representation with a single hidden layer and ReLU activation.
2. **Recurrent Neural Network (RNN)** — Processes word sequences using pretrained 50-dimensional word embeddings with a vanilla RNN (tanh nonlinearity).

Both models predict star ratings (1–5, mapped to labels 0–4) using LogSoftmax output and Negative Log-Likelihood loss.

## Repository Structure

```
.
├── ffnn.py              # Feed-Forward Neural Network implementation
├── rnn.py               # Recurrent Neural Network implementation
├── training.json        # Training data (8,000 examples)
├── validation.json      # Validation data (800 examples)
├── test.json            # Test data (800 examples)
├── word_embedding.pkl   # Pretrained 50-d word embeddings (for RNN)
├── CS4375_Report.pdf    # Writeup with results and analysis
└── README.md
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
conda create -n cs4375 python=3.8
conda activate cs4375
pip install torch numpy tqdm
```

> PyTorch 1.10.1 was benchmarked for this assignment. Any recent version should work.

## Usage

### FFNN

```bash
python ffnn.py --hidden_dim 10 --epochs 5 \
    --train_data training.json --val_data validation.json \
    --test_data test.json --do_train
```

### RNN

```bash
python rnn.py --hidden_dim 32 --epochs 10 \
    --train_data training.json --val_data validation.json \
    --test_data test.json
```

## Implementation

The core task was completing the `forward()` method in each model class.

**FFNN** — Three lines implementing `input → W1 → ReLU → W2 → LogSoftmax`:

```python
hidden = self.activation(self.W1(input_vector))
output = self.W2(hidden)
predicted_vector = self.softmax(output)
```

**RNN** — Four lines implementing `input sequence → RNN → linear projection → sum over time → LogSoftmax`:

```python
output, hidden = self.rnn(inputs)
output = self.W(output)
output = torch.sum(output, dim=0)
predicted_vector = self.softmax(output)
```

Test set evaluation was also added to both scripts (not present in the starter code).

## Results

| Model | Training Acc. | Validation Acc. | Test Acc. |
|-------|:------------:|:---------------:|:---------:|
| FFNN  | 64.08%       | 59.63%          | 11.38%    |
| RNN   | 51.94%       | 55.13%          | 9.88%     |

**Note on test accuracy:** The training set contains only stars 1–3, while the test set contains only stars 3–5. This label distribution mismatch means 80% of test examples belong to classes never seen during training, fully explaining the low test scores. See the report for detailed error analysis.

## Key Findings

- The FFNN outperforms the RNN on in-distribution data (59.63% vs. 55.13% validation accuracy), likely because bag-of-words captures strong lexical sentiment signals effectively on this small dataset.
- The RNN trains ~13x slower per epoch (~37s vs. ~2.9s) due to sequential processing.
- The vanilla RNN's performance is limited by vanishing gradients; an LSTM/GRU could improve results.
- The label distribution shift between train and test sets is the dominant factor in test performance.