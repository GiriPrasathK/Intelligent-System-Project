
import csv
import matplotlib.pyplot as plt

with open("Week11/Results/week11_scalability_results.csv", newline="") as f:
    data = list(csv.DictReader(f))

sizes = [int(r["Matrix_Size"]) for r in data]

plt.figure(figsize=(9, 6))

for name, col in [
    ("CPU", "CPU_ms"),
    ("GTX 1080", "GTX1080_ms"),
    ("Tesla T4", "T4_ms")
]:
    times = [float(r[col]) for r in data]
    plt.plot(sizes, times, marker="o", label=name)

plt.yscale("log")
plt.xlabel("Matrix Size")
plt.ylabel("Execution Time (ms) - Log Scale")
plt.title("CG Execution Time Scalability (Logarithmic Scale)")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()

plt.savefig("Week11/Plots/execution_time_scalability_log.png")
plt.show()

print("Logarithmic scalability graph generated successfully.")
