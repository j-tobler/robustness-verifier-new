import pandas as pd
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} csv_file upper_bound_txt_file output_pdf_file")
    sys.exit(1)

csv_file=sys.argv[1]
upper_bound_txt_file=sys.argv[2]
pdf_file=sys.argv[3]

upper_bound=0.0
with open(upper_bound_txt_file, 'r') as f:
    upper_bound=float(f.read().strip())

file_path = csv_file
data = pd.read_csv(file_path)

x = data.iloc[:, 0]  # First column for x-values
y = data.iloc[:, 1]  # Second column for y-values
y2 = data.iloc[:, 2]  # Third column for the second y-axis

fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(x, y, label="Certified Robustness", color="blue")
ax1.set_ylim(0.0,1.0)
ax1.set_xlabel("Gram Iterations")
ax1.set_ylabel("Robustness Proportion")
ax1.tick_params(axis='y', labelcolor="blue")

ax1.axhline(upper_bound, color="blue", linestyle="--", label=f"Measured Robustness")


ax2 = ax1.twinx()
ax2.plot(x, y2, label="Bounds Computation Time", color="green")
ax2.set_ylabel("Running Time (seconds)", color="green")
ax2.tick_params(axis='y', labelcolor="green")

plt.title("Robustness vs Gram Iterations (7x7 MNIST)")
ax2.legend(loc="upper right")
ax1.legend(loc="upper left")

plt.grid()

plt.savefig(pdf_file, format="pdf", bbox_inches="tight")

# Show the plot
#plt.show()
