import json

with open("/Users/test/Desktop/DTA400/results/dataset1_mc_summary.json", "r", encoding="utf-8") as f:
    summary = json.load(f)

print(json.dumps(summary, indent=2))
