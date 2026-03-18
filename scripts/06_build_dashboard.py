import json
from pathlib import Path
import re

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / 'outputs'
HTML_FILE = PROJECT_DIR / 'index.html'

def build_dashboard():
    # Load Metrics
    try:
        with open(OUTPUTS_DIR / 'web_mining_metrics.json', 'r') as f:
            web_metrics = json.load(f)
    except FileNotFoundError:
        print("Error: Run 04_web_mining.py first.")
        return

    req_per_hour = web_metrics.get("Peak Hour Requests", 60)
    latency_ms = web_metrics.get("Peak Hour Latency (ms)", 336.47)
    success_rate = web_metrics.get("Success Rate (%)", 80.0)
    error_rate = round(100.0 - success_rate, 2)
    clusters_found = web_metrics.get("Total Clusters Found", 3)

    # Read HTML
    with open(HTML_FILE, 'r') as f:
        html = f.read()

    # 1. Update API Traffic (Requests per min/hour)
    # The original says 140 req/min. We'll replace it with the peak traffic.
    html = re.sub(
        r'<span class="text-xl font-mono font-bold text-primary">\d+</span>\s*<span class="text-\[10px\] font-mono text-slate-600">req/min</span>',
        f'<span class="text-xl font-mono font-bold text-primary">{req_per_hour}</span>\n<span class="text-[10px] font-mono text-slate-600">req/hr (PEAK)</span>',
        html
    )

    # 2. Update Latency
    html = re.sub(
        r'<span class="text-xl font-mono font-bold text-primary">\d+</span>\s*<span class="text-\[10px\] font-mono text-slate-600">ms</span>',
        f'<span class="text-xl font-mono font-bold text-primary">{latency_ms}</span>\n<span class="text-[10px] font-mono text-slate-600">ms (PEAK)</span>',
        html
    )

    # 3. Update Error Rate
    # The original error rate uses emerald-muted for good status, but 20% error rate in peak is a warning (crimson)
    error_color = "emerald-muted" if error_rate < 5 else "crimson"
    html = re.sub(
        r'<span class="text-xl font-mono font-bold text-[^"]+">[\d.]+</span>\s*<span class="text-\[10px\] font-mono text-slate-600">%</span>',
        f'<span class="text-xl font-mono font-bold text-{error_color}">{error_rate}</span>\n<span class="text-[10px] font-mono text-slate-600">%</span>',
        html
    )

    # 4. Update the bottom logs
    log_text = (
        f'<span class="text-slate-500">[SYSTEM]</span> Extracted API Web Metrics... <br/>\n'
        f'<span class="text-slate-500">[SYSTEM]</span> Detected {clusters_found} distinct clinic geolocation clusters. <br/>\n'
        f'<span class="text-slate-500">[WARNING]</span> Load balancer recommended for Peak Hour {web_metrics.get("Peak Hour", 12)}:00.'
    )
    
    html = re.sub(
        r'<div class="w-1/3 font-mono text-\[10px\] text-primary/60 truncate bg-slate-900/50 p-3 rounded-lg border border-slate-800 self-center">.*?</div>',
        f'<div class="w-1/3 font-mono text-[10px] text-primary/60 truncate bg-slate-900/50 p-3 rounded-lg border border-slate-800 self-center">\n{log_text}\n</div>',
        html, flags=re.DOTALL
    )

    # Write back the HTML
    with open(HTML_FILE, 'w') as f:
        f.write(html)
        
    print("Dashboard Successfully Wired with Live Action Metrics!")

if __name__ == '__main__':
    build_dashboard()
