# Project 5: Scheduled Retraining with Evaluation Gate

Welcome to the automated Continuous Training (CT) pipeline! In this project, we use Prefect to orchestrate an end-to-end retraining job. Every week, it ingests new data, trains a candidate model, and mathematically ensures it doesn't degrade performance before promoting it.

## 📂 Project Structure

```text
📦 Scheduled Retraining Pipeline
 ┣ 📂 configs
 ┃ ┗ 📜 config.yaml         # Configuration for data paths and model settings
 ┣ 📂 src
 ┃ ┣ 📜 simulate_data.py    # Generates initial state and 3 weeks of test data
 ┃ ┗ 📜 pipeline.py         # The Prefect Flow (Ingest -> Train -> Evaluate Gate)
 ┣ 📜 RUNBOOK.md            # DevOps manual for intervening in the automated pipeline
 ┣ 📜 requirements.txt
 ┗ 📜 README.md
```

## 🚀 Quickstart Guide

### 1. Set up the environment

Create your virtual environment and install the required tools (Prefect, MLflow, Scikit-Learn):

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare the Data

Run the simulator to generate the historical master dataset, the strict validation set, and 3 simulated weeks of incoming data.

```bash
python src/simulate_data.py
```

### 3. Simulate the Automated Weeks

Run the Prefect pipeline manually to simulate time passing. Prefect will automatically format the terminal output to show you the pipeline steps running in real-time.

**Run Week 1 (Normal Data):**
Because this is the first run, there is no production model yet. It will automatically promote.

```bash
python src/pipeline.py --week data/incoming/week_1.csv
```

**Run Week 2 (Corrupted Data - The Safety Net Test):**
This week simulates a broken upstream data feed. The model will learn bad patterns. Watch the Evaluation Gate reject it!

```bash
python src/pipeline.py --week data/incoming/week_2_bad.csv
```

**Run Week 3 (Market Shift Data):**
This week simulates excellent new data capturing a change in the used car market. The model will learn it, beat the current production model, and get promoted!

```bash
python src/pipeline.py --week data/incoming/week_3_good.csv
```

### 4. View the Results in MLflow

To see a permanent record of all retrainings, metrics, and which ones got the `@production` alias, open the MLflow dashboard:

```bash
mlflow ui
```

Open http://127.0.0.1:5000 in your browser.
