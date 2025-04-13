import json
import matplotlib.pyplot as plt
import numpy as np

# Read the benchmark data
with open('benchmark_results.json', 'r') as f:
    data = json.load(f)

# Create a figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Single Request Latency vs Text Length
text_lengths = [data['single_request'][f'text_{i}']['text_length'] for i in range(1, 4)]
latencies = [data['single_request'][f'text_{i}']['mean'] for i in range(1, 4)]

ax1.plot(text_lengths, latencies, 'o-')
ax1.set_title('Single Request Latency vs Text Length')
ax1.set_xlabel('Text Length (characters)')
ax1.set_ylabel('Latency (seconds)')
ax1.grid(True)

# 2. Throughput vs Batch Size
batch_sizes = [1, 5, 10, 25, 50]
throughputs = [data['concurrent'][f'batch_{size}']['avg_throughput'] for size in batch_sizes]

ax2.plot(batch_sizes, throughputs, 'o-')
ax2.set_title('Throughput vs Batch Size')
ax2.set_xlabel('Batch Size')
ax2.set_ylabel('Throughput (requests/second)')
ax2.grid(True)

# 3. Latency Distribution for Different Batch Sizes
for size in batch_sizes:
    latencies = data['concurrent'][f'batch_{size}']['latencies']
    ax3.plot([size], [latencies['mean']], 'o', label=f'Batch {size}')
    ax3.errorbar(size, latencies['mean'], 
                yerr=[[latencies['mean'] - latencies['min']], 
                      [latencies['max'] - latencies['mean']]], 
                fmt='o', capsize=5)

ax3.set_title('Latency Distribution by Batch Size')
ax3.set_xlabel('Batch Size')
ax3.set_ylabel('Latency (seconds)')
ax3.grid(True)
ax3.legend()

# 4. Resource Usage
cpu_usage = data['resource_usage']['cpu']
memory_usage = data['resource_usage']['memory']

ax4.bar(['CPU', 'Memory'], 
        [cpu_usage['avg'], memory_usage['avg']],
        yerr=[[cpu_usage['avg'] - cpu_usage['min'], memory_usage['avg'] - memory_usage['min']],
              [cpu_usage['max'] - cpu_usage['avg'], memory_usage['max'] - memory_usage['avg']]],
        capsize=5)
ax4.set_title('Resource Usage')
ax4.set_ylabel('Usage (%)')
ax4.grid(True)

plt.tight_layout()
plt.savefig('benchmark_graphs.png')
plt.close() 