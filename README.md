# EMPA-ASA Hybrid-Cloud Scheduling — README

```

## Requirements
- **Python**: 3.10–3.12 (3.12 recommended)  
- **OS**: Windows / Linux / macOS (the research used Windows Server 2019 & Windows 10 as examples)  
- **Core packages**: `numpy`, `pandas`, `scipy`, `matplotlib`  
- **Optional**: `scikit-learn` (stats/plots), `gymnasium` (if you wrap an RL environment)

### Environment Setup
```bash
# 1) Create & activate a virtual environment (Windows PowerShell)
python -m venv .venv
. .venv/Scripts/Activate.ps1

# 2) Install dependencies
pip install numpy pandas scipy matplotlib
# Optional:
# pip install scikit-learn gymnasium
```
---

## Quick Start
Ensure a `data/` folder exists with:
- `tasks.json` — workload requests
- `vms.json` — compute nodes/VMs (capacities & costs)

Run the all-in-one comparison:
```bash
# from the project root (where 比较实验.py lives)
python 比较实验.py
```
Artifacts (CSV logs, figures) will be written to `results/` and/or a directory specified in the scripts.

---

## Data Files & Field Semantics

### 1) `tasks.json` (workload items)
Example entry:
```json
{
  "mem": 193,         // required memory (unit depends on your setup, e.g., MB)
  "con": 1,           // required compute concurrency/cores (logical units)
  "sto": 398,         // required storage capacity (MB/GB)
  "zone": "private",  // placement preference: private/public/any/cloud (per your experiment)
  "bandwidth": 2      // required bandwidth (e.g., MB/s)
}
```

### 2) `vms.json` (compute nodes / VMs / servers)
Example entry:
```json
{
  "mem": 1000,      // available memory
  "con": 4,         // available compute concurrency/cores
  "sto": 1000,      // available storage
  "zone": "private",// node location: private/public
  "cost_mem": 0.2,  // unit cost for memory
  "cost_con": 0.05, // unit cost for compute
  "cost_sto": 0.1,  // unit cost for storage
  "cost_bw": 0.05   // unit cost for bandwidth
}
```
> **Tip:** In your paper/report, explicitly state the units (MB/GB, cores, MB/s) and the billing period (per task/second/hour) for full reproducibility.

---

## How to Run

### A. Run an individual algorithm
```bash
python EMPA-ASA.py
python genetic_algorithm.py
python particle_swarm_optimization.py
python simulated_annealing.py
python traditional_eamp.py
```
Each script generally: loads `data/` → runs scheduling/optimization → computes QoS & cost metrics → writes results to `results/`.

### B. Unified comparison (recommended)

Typical pipeline:
1. Load `tasks.json` / `vms.json`.
2. Run EMPA‑ASA and baselines under **low/medium/high** load (or different `zone` constraints).
3. Produce **Response Time / Throughput / Latency / Jitter / Cost**.
4. Plot **2×2 figure** or separate bar charts (white background, publication style).


## Outputs & Visualization
- **CSV/JSON**: time‑stamped experiment logs with key QoS & cost fields.  
- **Figures**: recommended to save under `figures/`, for example:
  - `figures/rt_tp_latency_jitter_2x2.png` (300–600 DPI, white background)
  - `figures/cost_comparison_bar.png`
- **Logs**: record random seeds, Python & package versions, and machine configuration (CPU/RAM/OS).

---

## Reproducibility Tips
- Fix random seeds (`numpy.random.seed`, etc.).
- Export environment: `python -m pip freeze > requirements-lock.txt`.
- Document hardware: e.g., **private cloud Windows Server 2019 as the control node; public cloud Windows 10 hosting multiple virtual compute nodes** (specify CPU/RAM/disk/network in your paper’s “Experiments & Simulation” section).
- Ensure figures meet journal resolution guidelines (≥300 DPI, white background).
- For statistical tests (e.g., **Friedman**/**ANOVA**), publish repeated‑trial raw results under `results/` and provide the analysis code or link.

---

