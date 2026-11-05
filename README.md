# Blockudoku AI Agent

An efficient **Deep Q-Learning (DQN)** agent learning to play Blockudoku from scratch. Instead of the classic $Q(s, a)$ approach, the project utilizes an **Afterstate** ($\mathbf{V(s')}$) network that evaluates board states after a move, which drastically accelerates convergence and training stability.

### Key Features:
* **Afterstate Value Network ($V(s')$):** Evaluates the board based on value of post-action states.
* **Vectorized Environment:** Parallel simulation of multiple games simultaneously (batch inference) in pure NumPy/PyTorch.
* **ResNet Architecture:** Residual blocks supported by $1 \times 9$ and $9 \times 1$ filters (full-line analysis) and `nn.Embedding` for pieces.
* **(in development) ADB Deployment:** Agent is going to be able to test itself in real-world Bloku Doku game on an android device. (training games followed uniform distribution of pieces),

### Other experiments:
* **Smart Storage:** Stratified replay buffer (Stratified Sampling) that aimed to prevent early-game forgetting (*catastrophic forgetting*).
* **Auxiliary features:** Option of training NN agent also as aux features predictor, aimed to achieve explainable and stable learning process.
---

## Project Structure

```text
├── src/
│   ├── blokudoku_lib.py   # Game engine (VectorizedStates), action masks
│   ├── network_lib.py     # Neural network architectures and replay storages
│   ├── main.py            # Trainers and CLI entry point (`blokudoku` command)
│   ├── user_play.py       # Module allowing manual human gameplay
│   └── automatisation.py  # Automated play on android device (in development)
├── networks/              # Saved weights (*.pt) and their best scores (*.txt)
├── tests/                 # pytest suite (`uv run pytest`)
└── benchmarks/            # Speed comparisons, not part of the test suite
```

---
## Installation

Requires Python 3.10+ and the **uv** package manager (recommended) or standard `pip`.

```bash
git clone https://github.com/T-bultrowicz/BlockudokuAI.git
cd BlockudokuAI

# Install dependencies
uv sync
# or: pip install -r requirements.txt
```

---

## Usage

The project is operated via the command line interface:

* **Train / test the agent:**
  ```bash
  uv run blokudoku [train / test] [NN_type (casual / heavy / light / advanced / action)]
  ```
* **Manual play (compare score with AI):**
  ```bash
  uv run blokudoku play_as_player
  ```
* **Help and available options:**
  ```bash
  uv run blokudoku usage
  ```

---

## ADB Automation

In development

---

## Results

| Development stage | Average Score | Model Behaviour |
| :--- | :--- | :--- |
| DQN Classic Q(s,a) | ~300 pts | Baseline. Greedy local choices, lacking spatial planning. |
| CNN board transformation | ~600 pts | Minor score boost, progress still stuck due to large action space. |
| V1 Afterstate | **~2400 pts!** | First major breakthrough! Significant tactical improvement, however suffered from catastrophic forgetting. |
| Stratify and Aux features (V2) | ~1500 pts | Regression. Stratified sampling altered in-game state distribution, while auxiliary features disrupted goals of training. |
| Different sized architectures | ~1500 pts | Regression. Smaller models lacked representation capacity, while larger ones overfitted to replay buffer. |
| Breakthrough - scheduler | **~3950 pts!** | Learning rate decay stabilized gradients in further steps of learning, allowing the model to overcome the forgetting problem and further polish long-term strategies. |

---

## Author

Project created by **[T-bultrowicz](https://github.com/T-bultrowicz)** as part of experiments with advanced Reinforcement Learning techniques in surprisingly challenging practical application. 