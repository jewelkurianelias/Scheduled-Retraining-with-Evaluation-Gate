import argparse
import yaml
import pandas as pd
import mlflow
from prefect import flow, task
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load Configuration
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

MLFLOW_MODEL_NAME = config["model"]["registered_name"]

@task(name="1. Ingest Weekly Data", log_prints=True)
def ingest_data(new_data_path: str):
    """Appends new weekly data to the master training dataset."""
    print(f"📥 Ingesting new data from {new_data_path}...")
    master_df = pd.read_csv(config["data"]["master_dataset"])
    new_df = pd.read_csv(new_data_path)
    
    # Append and save back to master
    updated_master = pd.concat([master_df, new_df], ignore_index=True)
    updated_master.to_csv(config["data"]["master_dataset"], index=False)
    print(f"✅ Master dataset updated. New row count: {len(updated_master)}")
    return config["data"]["master_dataset"]

@task(name="2. Train Candidate Model", log_prints=True)
def train_model(train_data_path: str):
    """Trains a new model on the updated master dataset and logs to MLflow."""
    print("🧠 Training new candidate model...")
    df = pd.read_csv(train_data_path)
    X = df[config["model"]["features"]]
    y = df[config["model"]["target_column"]]
    
    mlflow.set_experiment("weekly_car_price_retraining")
    with mlflow.start_run() as run:
        model = RandomForestRegressor(
            n_estimators=100, 
            random_state=config["model"]["random_state"]
        )
        model.fit(X, y)
        
        # Log to MLflow
        mlflow.sklearn.log_model(model, "model")
        
        # Register the model version
        result = mlflow.register_model(f"runs:/{run.info.run_id}/model", MLFLOW_MODEL_NAME)
        
        print(f"✅ Model trained and registered as Version {result.version}.")
        return result.version

@task(name="3. Evaluate and Promote Gate", log_prints=True)
def evaluate_and_promote(candidate_version: str):
    """Compares candidate against production on a fixed benchmark. Promotes if better."""
    print(f"⚖️ Evaluating Candidate Version {candidate_version} against @production...")
    client = mlflow.MlflowClient()
    val_df = pd.read_csv(config["data"]["validation_dataset"])
    X_val = val_df[config["model"]["features"]]
    y_val = val_df[config["model"]["target_column"]]
    
    # Load and score Candidate
    candidate_uri = f"models:/{MLFLOW_MODEL_NAME}/{candidate_version}"
    candidate_model = mlflow.sklearn.load_model(candidate_uri)
    candidate_mae = mean_absolute_error(y_val, candidate_model.predict(X_val))
    print(f"↳ Candidate MAE: ${candidate_mae:.2f}")
    
    # Load and score Production (if it exists)
    try:
        prod_uri = f"models:/{MLFLOW_MODEL_NAME}@production"
        prod_model = mlflow.sklearn.load_model(prod_uri)
        prod_mae = mean_absolute_error(y_val, prod_model.predict(X_val))
        print(f"↳ Current Production MAE: ${prod_mae:.2f}")
    except Exception:
        print("↳ No @production model exists yet. Defaulting threshold to infinity.")
        prod_mae = float('inf')
    
    # The Promotion Gate Logic
    if candidate_mae < prod_mae:
        print(f"🏆 Candidate improved MAE by ${(prod_mae - candidate_mae):.2f}! Promoting...")
        client.set_registered_model_alias(MLFLOW_MODEL_NAME, "production", candidate_version)
        print("✅ SUCCESS: Model deployed to @production.")
    else:
        print(f"🛑 REJECTED: Candidate is worse or equal to production (Diff: +${(candidate_mae - prod_mae):.2f}).")
        print("↳ Pipeline complete. The bad model was blocked.")

@flow(name="Scheduled Retraining Pipeline")
def run_weekly_pipeline(week_file_path: str):
    """The master orchestrator flow that runs the entire sequence."""
    updated_data_path = ingest_data(week_file_path)
    candidate_version = train_model(updated_data_path)
    evaluate_and_promote(candidate_version)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=str, required=True, help="Path to the incoming weekly CSV")
    args = parser.parse_args()
    
    run_weekly_pipeline(args.week)