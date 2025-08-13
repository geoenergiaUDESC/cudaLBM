# cudaLBM

`cudaLBM` is a high-performance computing project that implements the moment representation of the Lattice Boltzmann Method (LBM) on a single Nvidia GPU using CUDA. This project is currently under development and is primarily focused on Linux-based systems.

## 🚀 Features

* **Lattice Boltzmann Method:** Implements the LBM for fluid dynamics simulations.

* **D3Q19 and D3Q27 Models:** Utilizes the D3Q19 velocity set, with plans to implement the D3Q27 model for more complex three-dimensional simulations.

* **High-Order Collision Models:** Will implement high-order collision models for improved accuracy.

* **Example Cases:** Includes a lid-driven cavity case to demonstrate the solver's capabilities, with more cases to be added in the future.

## ⚡ Performance

The code has been benchmarked with the lid driven cavity case on an **NVIDIA RTX A4000** and achieves approximately **3300 MLUPS** (Million Lattice Updates Per Second) using FP32 for both storage and arithmetic. There is still room for improvement, as further optimizations are planned.

## 📅 Future Features

* **Multi-GPU Support:** Future versions will leverage CUDA-aware MPI for efficient scaling across multiple GPUs.

## 🔧 Getting Started

### Prerequisites

* A C++ compiler (e.g., GCC)

* NVIDIA CUDA Toolkit

### Installation

1. Clone the repository:

```

git clone https://github.com/geoenergiaUDESC/cudaLBM.git

```

2. Navigate to the project directory and load the bashrc file:

```

cd cudaLBM
source bashrc

```

3. Compile the project:

```

make install

```

## 💨 Usage

To run a simulation, you can execute the compiled binary. For example, to run the lid-driven cavity case, navigate to the lidDrivenCavity folder and type:

```

momentBasedD3Q19 -GPU 0

```

## 📄 License

This project is licensed under the terms of the LICENSE file.

## 🙏 Acknowledgments

This codebase was heavily influenced by the `MR-LBM` project, although it has been completely rewritten. You can find the original repository here:

<https://github.com/CERNN/MR-LBM>
