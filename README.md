# NEUROSPIKE 🧠⚡

A brain-inspired learning framework combining Spiking Neural Networks (SNNs), computational neuroscience, and hardware design to build adaptive, low-power intelligent systems.

## Overview

NEUROSPIKE bridges neuroscience, machine learning, and neuromorphic computing by implementing **predictive coding** - a brain-inspired learning framework where the brain constantly predicts sensory input and learns from prediction errors. Our project:
- Simulates hierarchical predictive coding networks using biologically plausible spiking neurons in Brian2
- Implements LIF (Leaky Integrate-and-Fire) neuron circuits on FPGAs with Verilog for real-time predictive processing
- Explores energy-efficient, hardware-accelerated neural computation inspired by cortical microcircuits

### What Makes NEUROSPIKE Unique?

Unlike traditional deep learning approaches, predictive coding offers:
- **Biological Plausibility**: Mirrors how the brain processes information through bidirectional prediction and error signals
- **Local Learning Rules**: Neurons learn using only locally available information, enabling distributed computation
- **Energy Efficiency**: Sparse spiking activity reduces computational overhead, ideal for neuromorphic hardware
- **Hierarchical Inference**: Multi-level representations emerge naturally through prediction error minimization

## Team

- **Project Lead**: Manoj N H, EE-23b, IIT Madras
- **Faculty Advisor**: Gopalakrishnan Srinivasan, CSE Professor, IIT Madras
- **Team Members**: 4 Project members (Aryan, Avisha, Keshav, Natesan)

---

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Conda or Miniconda
- Git
- (Optional) Vivado/Quartus for FPGA development

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/neurospike.git
cd neurospike
```

#### 2. Create Conda Environment

```bash
# Create a new conda environment named 'neurospike'
conda create -n neurospike python=3.9 -y

# Activate the environment
conda activate neurospike
```

#### 3. Install Dependencies

```bash
# Install core dependencies
pip install brian2 numpy matplotlib scipy

# Install additional ML/data science tools
pip install pandas scikit-learn jupyter

# Install visualization tools
pip install seaborn plotly

# (Optional) Install PyTorch for hybrid SNN-DNN experiments
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
neurospike/
├── simulations/          # Brian2 simulation scripts
│   ├── lif_models/       # LIF neuron implementations
│   ├── predictive_coding/# Predictive coding models
│   └── networks/         # Network architectures
├── hardware/             # FPGA/hardware implementations
│   ├── verilog/          # Verilog HDL code
│   ├── testbenches/      # Simulation testbenches
│   └── constraints/      # FPGA constraint files
├── analysis/             # Data analysis and visualization
├── docs/                 # Documentation
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Usage

### Running Simulations

```bash
# Activate environment
conda activate neurospike

# Run basic LIF neuron simulation
python simulations/lif_models/basic_lif.py

# Run predictive coding network
python simulations/predictive_coding/pc_network.py
```

### Hardware Synthesis

```bash
# Navigate to hardware directory
cd hardware/verilog

# Simulate using your preferred tool (example with Icarus Verilog)
iverilog -o lif_neuron.vvp lif_neuron.v testbenches/lif_tb.v
vvp lif_neuron.vvp
```

### Jupyter Notebooks

```bash
# Start Jupyter Lab
conda activate neurospike
jupyter lab

# Navigate to analysis/ or simulations/ folders
```

---

## Key Features

### Software Simulations
- **Brian2-based SNN models**: Biologically realistic neuron dynamics
- **Predictive coding frameworks**: Hierarchical learning models
- **Network analysis tools**: Spike raster plots, firing rates, connectivity visualization

### Hardware Implementation
- **LIF neuron circuits**: Verilog implementations for FPGA deployment
- **Low-power design**: Optimized for energy efficiency
- **Real-time processing**: Hardware-accelerated computation

---

## Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
pytest tests/

# Format code
black .

# Lint code
flake8 simulations/ hardware/
```

### Contributing

1. Create a new branch for your feature
2. Make your changes
3. Run tests and ensure code quality
4. Submit a pull request

---

## Documentation

- [Brian2 Documentation](https://brian2.readthedocs.io/)
- [Spiking Neural Networks Tutorial](docs/snn_tutorial.md)
- [Hardware Implementation Guide](docs/hardware_guide.md)
- [API Reference](docs/api_reference.md)

---

## Useful Commands

```bash
# List conda environments
conda env list

# Deactivate environment
conda deactivate

# Update all packages
pip install --upgrade -r requirements.txt

# Export environment
conda env export > environment.yml

# Create environment from yml
conda env create -f environment.yml

# Remove environment
conda remove -n neurospike --all
```

---

## Research Goals

1. Develop biologically plausible learning algorithms
2. Implement energy-efficient neuromorphic hardware
3. Bridge computational neuroscience and machine learning
4. Validate models on benchmark tasks

---

## License

[MIT License](LICENSE)

---

## Contact

For questions or collaboration opportunities, please contact:
- Project Lead: [ee24b044@smail.iitm.ac.in]
- Repository: [https://github.com/a-jacked-nerd/neurospike](https://github.com/a-jacked-nerd/neurospike)

---

**Built with 🧠 at IIT Madras**
