# MCTS_DataGen

MCTS_DataGen is a tool for generating and processing Monte Carlo Tree Search (MCTS) data.

## Project Description

Briefly describe the main functions and purpose of the project.

## Installation

Describe how to install and set up the project:

```bash
git clone http://code.agibot.com/agentx/mcts_datagen.git
cd mcts_datagen
conda create --name datagen python=3.10
conda activate dategen
pip install -r requirements.txt

# download dataset
python scripts/alfworld-download.py

# begin
python src/main.py
```

## Project Structure

The project is structured as follows:

- `src/`: Source code
  - `mcts/`: Core MCTS implementation
  - `llm/`: Language model interfaces
  - `environment/`: Environment implementations
  - `visualization/`: Visualization tools
  - `utils/`: Utility functions
  - `main.py`: Main execution script
- `log/`: Log files
- `config/`: Configuration files
- `docs/`: Project documentation

## Contributing

If you want to contribute to this project, please follow these steps:
1. Fork this repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

The license used for this project - please fill in according to the actual situation

## Contact

Your Name - Your Email

Project Link: http://code.agibot.com/agentx/mcts_datagen
