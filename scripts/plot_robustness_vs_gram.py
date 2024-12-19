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

plt.figure(figsize=(8, 6))
plt.plot(x, y, label="Certified Robustness", color="blue")

plt.axhline(upper_bound, color="red", linestyle="--", label=f"Measured Robustness")

plt.ylim(0.0,1.0)

# Add labels, legend, and title
plt.xlabel("Gram Iterations")  # Replace with your actual label
plt.ylabel("Robustness")  # Replace with your actual label
plt.title("Robustness vs Gram Iterations (7x7 MNIST)")
plt.legend()
plt.grid()

plt.savefig(pdf_file, format="pdf", bbox_inches="tight")

# Show the plot
#plt.show()
