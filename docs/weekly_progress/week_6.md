# Week 6 – Development of a Machine Learning Application Using GPU Acceleration

## 1. Objective

The objective of Week 6 was to develop a machine learning application that uses GPU acceleration in the context of the Conjugate Gradient (CG) project.

The application predicts the execution time of the Conjugate Gradient algorithm based on matrix size and hardware configuration. PyTorch was used to develop and train a regression model using GPU acceleration.

## 2. Experimental Environment

| Component | Configuration |
|---|---|
| Platform | Google Colab |
| GPU | NVIDIA Tesla T4 |
| GPU Memory | 14.56 GB |
| Programming Language | Python |
| ML Framework | PyTorch 2.14.1 |
| CUDA | 13.0 |
| Numerical Computing | NumPy, SciPy |
| GPU Programming | Numba CUDA |

## 3. Dataset Generation

Sparse Symmetric Positive Definite (SPD) matrices were generated using a tridiagonal matrix structure.

The matrix was represented using Compressed Sparse Row (CSR) format.

Five matrix sizes were considered:

| Matrix Size | Non-zero Elements |
|---:|---:|
| 1,000 | 2,998 |
| 5,000 | 14,998 |
| 10,000 | 29,998 |
| 20,000 | 59,998 |
| 50,000 | 149,998 |

The right-hand-side vector was generated using a known solution:

\[
b=Ax_{\text{true}}
\]

This allowed numerical validation of the CG solutions.

## 4. CPU and GPU Benchmarking

The existing CPU and Numba CUDA-based GPU implementations of the Conjugate Gradient algorithm were reused.

Both implementations were executed using a tolerance of \(10^{-6}\).

The GPU implementation used:
- CSR sparse matrix-vector multiplication.
- GPU dot-product reduction using shared memory.
- GPU vector operations.
- Float64 arithmetic.
- True residual verification and residual replacement.

All five GPU benchmark cases converged successfully.

The GPU implementation also demonstrated numerical accuracy, with relative residuals below the specified tolerance.

The initial benchmark measurements showed that the CPU performed better for smaller matrices, while the Tesla T4 was faster for the 50,000-element problem in that measurement run.

Repeated measurements showed timing variation. Therefore, these results are preliminary and should not be treated as definitive hardware speedups.

## 5. Dataset Preparation

A combined dataset was prepared using CPU and Tesla T4 measurements.

The input features were:

- Matrix size.
- Hardware code (CPU = 0, Tesla T4 = 1).

The target variable was CG execution time in milliseconds.

Logarithmic transformations were applied to matrix size and execution time to handle their wide numerical ranges.

## 6. Machine Learning Model

A Feedforward Neural Network (Multilayer Perceptron) was implemented using PyTorch.

The architecture consisted of:

| Layer | Configuration |
|---|---|
| Input | 2 features |
| Hidden layer 1 | 32 neurons, ReLU |
| Hidden layer 2 | 16 neurons, ReLU |
| Output | 1 neuron |

Training configuration:

- Optimizer: Adam
- Learning rate: 0.001
- Loss function: Mean Squared Error (MSE)
- Training epochs: 1,000
- Training device: NVIDIA Tesla T4

The model was successfully trained with its parameters and input data placed on the GPU.

## 7. Model Evaluation

The model was evaluated using Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE).

### Combined model

| Hardware | MAE | MAPE |
|---|---:|---:|
| CPU | 3,545.124 ms | 124.65% |
| Tesla T4 | 548.593 ms | 13.46% |

### Hardware-specific model experiment

Separate regression models were also developed for CPU and Tesla T4.

| Hardware | MAE | MAPE |
|---|---:|---:|
| CPU | 3,723.679 ms | 156.01% |
| Tesla T4 | 1,002.651 ms | 21.07% |

The combined model achieved lower training-set errors than the separate models in these experiments.

These metrics represent fit on the training observations, not independent test performance.

## 8. Application Development

An interactive CG Performance Prediction application was developed using Python and ipywidgets.

The application provides:

1. Matrix size selection.
2. Hardware selection.
3. Predicted CG execution time.
4. Comparison of predicted CPU and GPU execution times.

The trained PyTorch model was saved for subsequent use.

## 9. Observations

1. GPU acceleration was successfully demonstrated for machine learning model training.
2. The current CG implementation exhibited significant CPU–GPU synchronization overhead.
3. GPU performance was not necessarily better for smaller matrix sizes.
4. The combined regression model achieved lower training-set prediction errors than the separate hardware-specific models.
5. The small dataset limited the reliability of the prediction model.

## 10. Limitations and Future Improvements

The current dataset contains only five observations per hardware type.

Future improvements include:

- Repeated benchmark measurements.
- Additional sparse matrix configurations.
- Incorporation of NVIDIA RTX A6000 workstation measurements.
- Proper training, validation and independent test datasets.
- Evaluation of additional regression algorithms.
- Reduction of CPU–GPU synchronization overhead in the CG implementation.

## 11. Conclusion

Week 6 successfully demonstrated the development of a machine learning application using GPU acceleration in the context of the Conjugate Gradient project.

A PyTorch-based regression model was trained on the NVIDIA Tesla T4 to predict CG execution time using matrix size and hardware type.

The work connected GPU computing, sparse numerical linear algebra and machine learning within a single application.

The resulting implementation provides a foundation for further performance analysis using the local RTX A6000 workstation and cloud GPU environments.
