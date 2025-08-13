# cudaLBM

`cudaLBM` is a high-performance computing project that implements the moment representation of the Lattice Boltzmann Method (LBM) on a single GPU using CUDA. This project is currently under development and is primarily focused on Linux-based systems.

## 🚀 Features

* **Lattice Boltzmann Method:** Implements the LBM for fluid dynamics simulations.

* **D3Q19 and D3Q27 Models:** Utilizes the D3Q19 velocity set, with plans to implement the D3Q27 model for more complex three-dimensional simulations.

* **High-Order Collision Models:** Will implement high-order collision models for improved accuracy.

* **Example Cases:** Includes a lid-driven cavity case to demonstrate the solver's capabilities, with more cases to be added in the future.

## 📅 Future Features

* **Multi-GPU Support:** Future versions will leverage CUDA-aware MPI for efficient scaling across multiple GPUs.

## 🔧 Getting Started

### Prerequisites

* A C++ compiler (e.g., GCC)

* NVIDIA CUDA Toolkit

### Installation

1. Clone the repository:

```

git clone [https://github.com/geoenergiaUDESC/cudaLBM.git](https://github.com/geoenergiaUDESC/cudaLBM.git)

```

2. Navigate to the project directory:

```

cd cudaLBM

```

3. Compile the project:

```

make

```

## 💨 Usage

To run a simulation, you can execute the compiled binary. For example, to run the lid-driven cavity case:

```

./cudaLBM cases/lidDrivenCavity/input.dat

```

## 🤝 Contributing

Contributions are welcome! If you would like to contribute to this project, please follow these steps:

1. Fork the repository.

2. Create a new branch (`git checkout -b feature/your-feature`).

3. Commit your changes (`git commit -am 'Add some feature'`).

4. Push to the branch (`git push origin feature/your-feature`).

5. Create a new Pull Request.

## 📄 License

This project is licensed under the terms of the LICENSE file.

## 🙏 Acknowledgments

This codebase was heavily influenced by the `MR-LBM` project, although it has been completely rewritten. You can find the original repository here:

<https://github.com/CERNN/MR-LBM>
```
