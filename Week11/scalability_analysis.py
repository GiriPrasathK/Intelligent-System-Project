import csv
import os
import matplotlib.pyplot as plt

INPUT_FILE = "Notebooks/repeated_benchmark_comparison.csv"
RESULTS_DIR = "Week11/Results"
PLOTS_DIR = "Week11/Plots"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Read existing benchmark results
data = []

with open(INPUT_FILE, newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        data.append({
            "Matrix_Size": int(row["Matrix_Size"]),
            "CPU_ms": float(row["CPU_ms"]),
            "GTX1080_ms": float(row["GTX1080_ms"]),
            "T4_ms": float(row["T4_ms"]),
            "GTX1080_Speedup": float(row["GTX1080_Speedup"]),
            "T4_Speedup": float(row["T4_Speedup"])
        })

# Save scalability analysis results
output_file = os.path.join(RESULTS_DIR, "week11_scalability_results.csv")

with open(output_file, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)

sizes = [row["Matrix_Size"] for row in data]

# Plot 1: Execution time comparison
plt.figure(figsize=(9, 6))

for environment, column in [
    ("CPU", "CPU_ms"),
    ("GTX 1080", "GTX1080_ms"),
    ("Tesla T4", "T4_ms")
]:
    plt.plot(
        sizes,
        [row[column] for row in data],
        marker="o",
        label=environment
    )

plt.xlabel("Matrix Size")
plt.ylabel("Execution Time (ms)")
plt.title("CG Execution Time vs Matrix Size")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "execution_time_scalability.png"))
plt.close()

# Plot 2: GPU speedup comparison
plt.figure(figsize=(9, 6))

plt.plot(
    sizes,
    [row["GTX1080_Speedup"] for row in data],
    marker="o",
    label="GTX 1080"
)

plt.plot(
    sizes,
    [row["T4_Speedup"] for row in data],
    marker="o",
    label="Tesla T4"
)

plt.axhline(y=1, linestyle="--", label="CPU performance")
plt.xlabel("Matrix Size")
plt.ylabel("Speedup over CPU")
plt.title("GPU Speedup vs Matrix Size")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "gpu_speedup_scalability.png"))
plt.close()

print("Week 11 scalability analysis completed.")
print(f"Results saved: {output_file}")
print(f"Plots saved in: {PLOTS_DIR}")
