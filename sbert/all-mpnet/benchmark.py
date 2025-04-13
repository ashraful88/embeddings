import requests
import time
import statistics
import json
import concurrent.futures
import psutil
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
API_URL = "http://localhost:8000/embed"
TEST_TEXTS = [
    "Short text for embedding.",
    "This is a medium length text that contains multiple words and should be processed by the embedding model.",
    "This is a longer piece of text that would represent a more substantial paragraph that might be processed in a real-world scenario. It contains multiple sentences and should give a good indication of performance for longer inputs."
]
BATCH_SIZES = [1, 5, 10, 25, 50]  # Number of concurrent requests
ITERATIONS = 10  # Number of measurement iterations per configuration

results = {
    "single_request": {},
    "concurrent": {},
    "resource_usage": {}
}

# 1. Single request latency for different text lengths
print("Testing single request latency...")
for i, text in enumerate(TEST_TEXTS):
    latencies = []
    for _ in range(ITERATIONS):
        start_time = time.time()
        response = requests.post(API_URL, json={"text": text})
        if response.status_code == 200:
            latency = time.time() - start_time
            latencies.append(latency)
    
    results["single_request"][f"text_{i+1}"] = {
        "min": min(latencies),
        "max": max(latencies),
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "p95": sorted(latencies)[int(0.95 * len(latencies))],
        "text_length": len(text)
    }

# 2. Throughput testing with concurrent requests
print("Testing throughput with concurrent requests...")

def make_request(text):
    start_time = time.time()
    response = requests.post(API_URL, json={"text": text})
    return time.time() - start_time if response.status_code == 200 else None

for batch_size in BATCH_SIZES:
    throughput_results = []
    
    for _ in range(ITERATIONS):
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Use medium length text for throughput testing
            futures = [executor.submit(make_request, TEST_TEXTS[1]) for _ in range(batch_size)]
            latencies = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        elapsed = time.time() - start_time
        requests_per_second = batch_size / elapsed
        throughput_results.append(requests_per_second)
    
    results["concurrent"][f"batch_{batch_size}"] = {
        "min_throughput": min(throughput_results),
        "max_throughput": max(throughput_results),
        "avg_throughput": statistics.mean(throughput_results),
        "latencies": {
            "min": min(latencies),
            "max": max(latencies),
            "mean": statistics.mean(latencies),
            "p95": sorted(latencies)[int(0.95 * len(latencies))]
        }
    }

# 3. Resource Usage (CPU, RAM)
print("Measuring resource usage...")
# First warm up the service
requests.post(API_URL, json={"text": TEST_TEXTS[1]})

cpu_percentages = []
memory_usages = []

# Monitor during a medium load test
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = []
    for _ in range(50):  # Send 50 requests in batches of 10
        futures.append(executor.submit(make_request, TEST_TEXTS[1]))
        
        # Sample resource usage every few requests
        if len(futures) % 5 == 0:
            cpu_percentages.append(psutil.cpu_percent(interval=0.1))
            memory_usages.append(psutil.virtual_memory().percent)

results["resource_usage"] = {
    "cpu": {
        "min": min(cpu_percentages),
        "max": max(cpu_percentages),
        "avg": statistics.mean(cpu_percentages)
    },
    "memory": {
        "min": min(memory_usages),
        "max": max(memory_usages),
        "avg": statistics.mean(memory_usages)
    }
}

# Save results
with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Benchmark complete! Results saved to benchmark_results.json")

# Generate plots
os.makedirs("benchmark_graphs", exist_ok=True)

# 1. Text length vs latency
plt.figure(figsize=(10, 6))
text_lengths = [results["single_request"][f"text_{i+1}"]["text_length"] for i in range(len(TEST_TEXTS))]
latencies = [results["single_request"][f"text_{i+1}"]["mean"] for i in range(len(TEST_TEXTS))]
plt.plot(text_lengths, latencies, marker='o')
plt.xlabel("Text Length (characters)")
plt.ylabel("Latency (seconds)")
plt.title("Text Length vs. Latency")
plt.grid(True)
plt.savefig("benchmark_graphs/text_length_vs_latency.png")

# 2. Batch size vs throughput
plt.figure(figsize=(10, 6))
batch_sizes = BATCH_SIZES
throughputs = [results["concurrent"][f"batch_{size}"]["avg_throughput"] for size in BATCH_SIZES]
plt.plot(batch_sizes, throughputs, marker='o')
plt.xlabel("Concurrent Requests")
plt.ylabel("Throughput (requests/second)")
plt.title("Concurrency vs. Throughput")
plt.grid(True)
plt.savefig("benchmark_graphs/concurrency_vs_throughput.png")

# 3. CPU and Memory usage during load test
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(cpu_percentages)
plt.xlabel("Sample")
plt.ylabel("CPU Usage (%)")
plt.title("CPU Usage During Load Test")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(memory_usages)
plt.xlabel("Sample")
plt.ylabel("Memory Usage (%)")
plt.title("Memory Usage During Load Test")
plt.grid(True)

plt.tight_layout()
plt.savefig("benchmark_graphs/resource_usage.png")

print("Graphs generated in benchmark_graphs directory")