# HumanEvalComm V2 Leaderboard

[![Flask](https://img.shields.io/badge/Flask-2.3+-blue.svg)](https://flask.palletsprojects.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)

A clean, interactive dashboard for exploring HumanEvalComm V2 benchmark results. Perfect for researchers and developers analyzing code generation models.

![Dashboard Preview](https://via.placeholder.com/800x400/2563eb/ffffff?text=HumanEvalComm+V2+Dashboard)

## ✨ What it does

- **Smart Data Loading**: Automatically finds and loads your latest benchmark results
- **Interactive Charts**: Beautiful visualizations powered by Plotly
- **Detailed Analysis**: Deep-dive into individual model evaluations
- **Easy Filtering**: Sort and filter results by model, metrics, or problems
- **Export Options**: Download data as CSV for further analysis

## 📊 Features

### Main Dashboard

- Sortable leaderboard table with all your models
- Multiple interactive charts (bar charts, radar plots, heatmaps)
- Real-time data refresh
- Export results to CSV

### Detailed Analysis

- Click any result for in-depth analysis
- Tabbed interface: Overview, Code Analysis, Metrics, Technical Details
- View original problems, model responses, and extracted code
- Compare performance across different evaluation metrics

## 🚀 Quick Start

```bash
# Get the code
git clone https://github.com/your-username/humaneval-comm-leaderboard.git
cd humaneval-comm-leaderboard/flask_leaderboard

# Install and run
pip install -r requirements.txt
python app.py

# Visit http://localhost:8080
```

## 🛠️ Setup

### Requirements

- Python 3.8+
- Your benchmark data files:
  - `v2_*leaderboard*.csv` (main results)
  - `v2_*results*.json` (detailed evaluations)
  - `Benchmark/HumanEvalComm_v2.jsonl` (problem definitions)

### Installation

```bash
pip install flask pandas plotly
```

## 🎯 Usage

### Basic Navigation

- **Main Leaderboard**: Overview of all model results
- **Detailed Results**: Individual problem analysis
- **Charts**: Interactive visualizations
- **Export**: Download data for external analysis

### API Access

```python
import requests

# Get all data
data = requests.get('http://localhost:8080/api/data').json()

# Get chart data
charts = requests.get('http://localhost:8080/api/charts').json()
```

## ⚙️ Configuration

### Data Directory

By default, the leaderboard looks for benchmark results in the `benchmark_v2/` directory. You can customize this by setting the `HUMANEVAL_DATA_DIR` environment variable:

```bash
# Use a custom data directory
export HUMANEVAL_DATA_DIR="/path/to/your/benchmark/results"
python app.py

# Or run directly
HUMANEVAL_DATA_DIR="/path/to/your/benchmark/results" python app.py
```

The application expects these files in your data directory:

- `v2_*leaderboard*.csv` (main results table)
- `v2_*results*.json` (detailed evaluation data)

### Data Location

Place your files in the project root or modify `app_enhanced.py` to point to your data directory.

### UI Changes

Edit the HTML templates in the `templates/` folder to customize the interface.

## 🤝 Contributing

We welcome improvements! Here's how to help:

1. Fork the repository
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## 📄 License

MIT License - feel free to use this for your research projects.

---

**Built for the AI research community** ❤️
