import statistics
import os

# Traverse through all folders and extract 'test time for sample' values from time.txt files
base_path = r"E:\SLIIT RA 2024\SICET\zST-GCN_CWBG_LOOPnew\results_randomSplit\full"
sample_times = []

for root, dirs, files in os.walk(base_path):
    if "log" in root and "time.txt" in files:
        time_file_path = os.path.join(root, "time.txt")
        with open(time_file_path, "r") as file:
            for line in file:
                if "test time for sample:" in line:
                    time_value = line.split("test time for sample:")[1].strip()
                    sample_times.append(float(time_value))


# Compute mean and standard deviation
mean_time = statistics.mean(sample_times)
std_time = statistics.stdev(sample_times)

print(len(sample_times))
print(f"Mean sample test time: {mean_time:.10f} seconds")
print(f"Standard deviation:    {std_time:.10f} seconds")

