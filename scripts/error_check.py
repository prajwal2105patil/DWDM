import pandas as pd
import re

log_pattern = re.compile(
    r'(?P<ip>\d+\.\d+\.\d+\.\d+) - - \[(?P<time>[^\]]+)\] ".+" (?P<status>\d{3}) .+'
)

parsed = []
with open('data/api_logs.txt', 'r') as f:
    for line in f:
        match = log_pattern.match(line)
        if match:
            parsed.append(match.groupdict())

df = pd.DataFrame(parsed)
df['time'] = pd.to_datetime(df['time'], format='%d/%b/%Y:%H:%M:%S %z')
df['hour'] = df['time'].dt.hour
df['status'] = df['status'].astype(int)

# Total errors
err_df = df[df['status'].isin([400, 500])]

print('--- ERROR COUNT BY IP (ALL HOURS) ---')
print(err_df.groupby('ip').size().sort_values(ascending=False))

print('\n--- ERROR COUNT BY IP (AT 12:00 PM Peak) ---')
err_12 = err_df[err_df['hour'] == 12]
print(err_12.groupby('ip').size().sort_values(ascending=False))
