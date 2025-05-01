# Embodied Agent Planning Toolkit

This repository contains two complementary components for embodied agent planning:
1. `mcts-datagen`: Monte Carlo Tree Search data generation pipeline
2. `LLaMA-Factory`: Model training and evaluation framework

## Project Structure

```
Embodied-Agent-Planning/
├── LLaMA-Factory/ # Model training and evaluation
│ ├── run_scripts/ # Training and evaluation scripts
│ ├── src/ # Training pipeline source code
│ └── ...
├── mcts-datagen/ # MCTS data generation
│ ├── src/ # Core source code
│ │ ├── mcts/ # Core MCTS implementation
│ │ ├── llm/ # Language model interfaces
│ │ ├── environment/ # Environment implementations
│ │ ├── visualization/ # Visualization tools
│ │ ├── utils/ # Utility functions
│ │ └── main.py # Main execution script
│ ├── scripts/ # Data processing scripts
│ ├── log/ # Log files
│ ├── config/ # Configuration files
│ ├── docs/ # Project documentation
│ └── ...
└── README.md # This documentation
```

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/xuhang-2/Embodied-Agent-Planning.git
cd Embodied-Agent-Planning
```

### 2. Set up environment variables
Add these to your `.bashrc` or equivalent:
```bash
export PROJECT_ROOT=/path/to/Embodied-Agent-Planning
export MCTS_DATAGEN=$PROJECT_ROOT/mcts-datagen
export LLAMA_FACTORY=$PROJECT_ROOT/LLaMA-Factory
```

## MCTS Data Generation (mcts-datagen)

### Setup
```bash
cd $MCTS_DATAGEN
conda create --name mcts python=3.10
conda activate mcts
pip install -r requirements.txt

Download dataset
python scripts/alfworld-download.py
```

### Generate Data
```bash
python src/main.py
```

## Model Training (LLaMA-Factory)

### Setup
```bash
cd $LLAMA_FACTORY
conda create --name llama-factory python=3.10
conda activate llama-factory
pip install -r requirements.txt
```

### Training
```bash
Using provided script (modify paths in script first)
bash run_scripts/train/llama3_dpo_train.sh
```

### Evaluation
```bash
bash run_scripts/evaluation/llama3_dpo_eval.sh
```

## Configuration Notes

Before running scripts in `LLaMA-Factory/run_scripts/`, update these variables:
```bash
In train.sh and eval.sh:
PROJECT_ROOT=/path/to/Embodied-Agent-Planning
MCTS_DATAGEN=$PROJECT_ROOT/mcts-datagen
LLAMA_FACTORY=$PROJECT_ROOT/LLaMA-Factory
```

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

Your Name - Your Email  
Project Link: https://github.com/xuhang-2/Embodied-Agent-Planning