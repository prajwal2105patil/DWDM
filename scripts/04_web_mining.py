import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import re
from sklearn.cluster import KMeans
import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUTS_DIR = PROJECT_DIR / 'outputs'

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# 1. Generation: Create 500 simulated log entries
random.seed(42)
np.random.seed(42)

base_ips = ["192.168.1.100", "10.0.0.50", "172.16.0.25", "192.168.2.200", "10.1.1.10"]
status_codes = [200] * 80 + [400] * 10 + [500] * 10
logs = []

start_time = datetime(2023, 10, 1, 0, 0, 0)

# We will define the peak hours as 10-14, traffic is naturally higher then.
for i in range(500):
    hour_offset = int(np.random.normal(12, 4)) % 24
    log_time = start_time + timedelta(days=random.randint(0, 30), hours=hour_offset, minutes=random.randint(0, 59))
    ip = random.choice(base_ips)[:-1] + str(random.randint(0, 9))
    status = random.choice(status_codes)
    endpoint = "/api/v1/predict"
    
    # Latency: base latency ~50ms, but spikes significantly if between 11-13 hours (Peak Traffic)
    if 11 <= hour_offset <= 13:
        latency_ms = int(np.random.normal(350, 100)) # Bottleneck
    else:
        latency_ms = int(np.random.normal(60, 20)) # Normal
    
    latency_ms = max(10, latency_ms)

    log_line = f'{ip} - - [{log_time.strftime("%d/%b/%Y:%H:%M:%S +0000")}] "POST {endpoint} HTTP/1.1" {status} 1234 {latency_ms}'
    logs.append(log_line)

log_path = DATA_DIR / 'api_logs.txt'
with open(log_path, 'w') as f:
    f.writelines(line + '\n' for line in logs)

# 2. Extraction using regex/pandas
log_pattern = re.compile(
    r'(?P<ip>\d+\.\d+\.\d+\.\d+) - - \[(?P<time>[^\]]+)\] "(?P<request>[^"]+)" (?P<status>\d{3}) \d+ (?P<latency>\d+)'
)

parsed_data = []
for line in logs:
    match = log_pattern.match(line)
    if match:
        parsed_data.append(match.groupdict())

df_logs = pd.DataFrame(parsed_data)
df_logs['time'] = pd.to_datetime(df_logs['time'], format="%d/%b/%Y:%H:%M:%S %z")
df_logs['hour'] = df_logs['time'].dt.hour
df_logs['status'] = df_logs['status'].astype(int)
df_logs['latency'] = df_logs['latency'].astype(int)

hourly_density = df_logs['hour'].value_counts().sort_index().to_dict()
peak_hour = max(hourly_density, key=hourly_density.get)

# Calculate latency correlation: Average latency per hour
hourly_latency = df_logs.groupby('hour')['latency'].mean().to_dict()
peak_hour_latency = hourly_latency[peak_hour]
off_peak_latency = df_logs[df_logs['hour'] != peak_hour]['latency'].mean()

status_dist = df_logs['status'].value_counts().to_dict()
success_rate = status_dist.get(200, 0) / len(df_logs) * 100

def ip_to_int(ip):
    parts = list(map(int, ip.split('.')))
    return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]

df_logs['ip_int'] = df_logs['ip'].apply(ip_to_int)
X = df_logs['ip_int'].values.reshape(-1, 1)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_logs['clinic_cluster'] = kmeans.fit_predict(X)
cluster_counts = df_logs['clinic_cluster'].value_counts().to_dict()
most_active_cluster = max(cluster_counts, key=cluster_counts.get)

# Compile results
results = {
    "Peak Hour": int(peak_hour),
    "Peak Hour Requests": int(hourly_density[peak_hour]),
    "Peak Hour Latency (ms)": round(float(peak_hour_latency), 2),
    "Off-Peak Latency (ms)": round(float(off_peak_latency), 2),
    "Success Rate (%)": round(float(success_rate), 2),
    "Total Clusters Found": 3,
    "Most Active Cluster ID": int(most_active_cluster),
    "Hourly Density": hourly_density,
    "Status Distribution": status_dist
}

out_path = OUTPUTS_DIR / 'web_mining_metrics.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Web mining simulation complete. Metrics saved to {out_path}.")
