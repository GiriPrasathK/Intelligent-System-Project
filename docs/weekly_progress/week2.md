# Week 2 Project Work: Identifying a GPU-Suitable Workload

## 1. Selected Workload
The selected workload is the Conjugate Gradient (CG) method for solving large sparse symmetric positive-definite linear systems of the form Ax = b.

## 2. Computational Requirements
The CG method repeatedly performs sparse matrix-vector multiplication, vector updates, dot products, and convergence checks. These operations are repeated until the solution satisfies the specified convergence criterion.

## 3. Suitability for GPU Acceleration
Sparse matrix-vector multiplication and vector updates expose data parallelism because multiple output elements can often be computed independently. Dot products can also use parallel computation followed by a reduction.

However, irregular memory access, synchronization, data-transfer overhead, and dependencies between iterations can affect GPU performance.

## 4. Proposed Experimental Comparison
The project will investigate the CG workload across:
- CPU execution
- Local GPU execution
- Cloud GPU execution

The comparison will examine execution time, problem size, convergence accuracy, and computational overhead. Cloud cost and resource utilization will be measured where feasible.

## 5. Initial Research Questions
1. How does CG execution time change as matrix size increases?
2. Under what conditions does GPU execution outperform CPU execution?
3. How do memory access and data-transfer overhead affect performance?
4. What additional overheads arise when executing CG on a cloud GPU?

## 6. Conclusion
The CG method is a suitable candidate for studying CPU–GPU performance because it combines parallel numerical operations with iterative dependencies. Experimental measurements will be used to determine the actual performance differences.

## 7. Status
Week 2 project work: workload selected and its GPU suitability documented. Implementation and benchmarking remain pending.
