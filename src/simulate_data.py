import os
import pandas as pd
import numpy as np

def generate_car_data(n_samples, noise_multiplier=1.0, base_price=20000):
    """Generates synthetic used car pricing data."""
    np.random.seed(42)
    
    year = np.random.randint(2005, 2024, n_samples)
    mileage = np.random.randint(10000, 200000, n_samples)
    condition_score = np.random.randint(1, 11, n_samples)
    
    # Simple pricing logic: newer cars, lower mileage, better condition = higher price
    age = 2024 - year
    price = base_price - (age * 800) - (mileage * 0.05) + (condition_score * 500)
    
    # Add noise
    noise = np.random.normal(0, 1500 * noise_multiplier, n_samples)
    price = price + noise
    
    # Ensure no negative prices
    price = np.maximum(price, 1000)
    
    return pd.DataFrame({
        "year": year,
        "mileage": mileage,
        "condition_score": condition_score,
        "price": price.round(2)
    })

def setup_simulation():
    """Creates the initial project state and simulates 3 weeks of future data."""
    os.makedirs("data/incoming", exist_ok=True)
    
    print("🚗 Generating synthetic used car data...")
    
    # 1. Master Training Data (The historical data)
    master_df = generate_car_data(1000)
    master_df.to_csv("data/master_train.csv", index=False)
    
    # 2. Fixed Validation Data (The strict benchmark we evaluate all models against)
    val_df = generate_car_data(200)
    val_df.to_csv("data/fixed_validation.csv", index=False)
    
    # 3. Simulate Week 1 Data (Normal, helpful new data)
    week1_df = generate_car_data(200, noise_multiplier=0.9) # Slightly cleaner data
    week1_df.to_csv("data/incoming/week_1.csv", index=False)
    
    # 4. Simulate Week 2 Data (A bad week - corrupted data/heavy noise)
    week2_df = generate_car_data(200, noise_multiplier=5.0) # Terrible data
    week2_df.to_csv("data/incoming/week_2_bad.csv", index=False)
    
    # 5. Simulate Week 3 Data (Excellent new data capturing a market shift)
    week3_df = generate_car_data(300, noise_multiplier=0.5, base_price=22000) # Shifted market
    week3_df.to_csv("data/incoming/week_3_good.csv", index=False)
    
    print("✅ Setup complete! Created master data, validation set, and 3 simulated weeks of incoming data.")

if __name__ == "__main__":
    setup_simulation()