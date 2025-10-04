# NYC Yellow Taxi Data Analysis and Prediction Models

## Project Overview

This project analyzes the NYC Yellow Taxi dataset from 2023 to build predictive models for two key business metrics: **fare prediction** and **tip prediction**. The analysis includes comprehensive data cleaning, exploratory data analysis, feature engineering, and machine learning model development using scikit-learn pipelines.

## Business Context

For NYC taxi operators and drivers, understanding fare and tip patterns is crucial for:
- **Revenue optimization**: Predicting expected fares for different routes and times
- **Driver earnings**: Understanding tip patterns to maximize driver income
- **Operational planning**: Identifying high-value locations and time periods
- **Customer service**: Setting appropriate expectations for ride costs

## Dataset Description

The dataset contains NYC Yellow Taxi trip records from 2023, with each row representing a single taxi trip. Key attributes include:

- **Trip details**: Pickup/dropoff times and locations, trip distance, passenger count
- **Fare breakdown**: Base fare, extras, taxes, tolls, surcharges
- **Payment info**: Payment type, tip amount, total amount
- **Operational data**: Vendor ID, rate code, store-and-forward flag

## Project Structure

```
project-1-chanakyavasantha/
├── data/
│   ├── yellow_taxi_data.csv          # Raw dataset
│   ├── processed_data/
│   │   ├── preprocessing_pipeline.pkl  # Data preprocessing pipeline
│   │   └── processed_data.pkl          # Cleaned and processed data
│   └── splits/
│       ├── fare_train_test.pkl         # Train/test split for fare prediction
│       └── tip_train_test.pkl          # Train/test split for tip prediction
├── models/
│   ├── fare_prediction/
│   │   ├── linear_regression.pkl       # Trained linear regression model
│   │   └── lasso_regression.pkl        # Trained lasso regression model
│   └── tip_prediction/
│       ├── linear_regression.pkl       # Trained linear regression model
│       └── lasso_regression.pkl        # Trained lasso regression model
├── plots/
│   └── data_visualisation/
│       ├── correlation_heatmap.png     # Feature correlation analysis
│       ├── tip_distributions.png       # Tip amount distributions
│       ├── tip_distributions_grid.png  # Tip patterns by categories
│       └── top_locations_tips.png      # Geographic tip analysis
├── results/
│   ├── fare_model_comparison.csv       # Fare model performance metrics
│   ├── tip_model_comparison.csv        # Tip model performance metrics
│   └── figures/
│       ├── fare_prediction/
│       │   └── test_predictions.png    # Fare prediction results
│       └── tip_prediction/
│           ├── lasso_feature_importance.png
│           ├── lr_feature_importance.png
│           ├── test_predictions.png
│           └── test_residuals.png
├── training.ipynb                      # Model training and hyperparameter tuning
├── test.ipynb                         # Model evaluation on test set
└── README.md                          # This file
```

## Key Features and Engineering

The project implements several important feature engineering steps:

1. **Temporal Features**:
   - Day of week extraction (Monday through Sunday)
   - Time categorization: Morning (10:00-11:59), Afternoon (12:00-16:59), Evening (17:00-18:59), Night (19:00-20:59)

2. **Derived Features**:
   - `pre_tip_total_amount`: Sum of all fare components before tip (fare_amount + extra + mta_tax + tolls_amount + improvement_surcharge + congestion_surcharge + airport_fee)

3. **Data Cleaning**:
   - Outlier detection and removal
   - Missing value handling
   - Data type optimization

## Model Performance

### Fare Prediction Models
| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Linear Regression | 0.926 | 4.83 | 2.83 |
| Lasso Regression | 0.927 | 4.82 | 2.84 |

### Tip Prediction Models
| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Linear Regression | 0.563 | 2.54 | 1.51 |
| Lasso Regression | 0.564 | 2.54 | 1.50 |

**Key Insights**:
- Fare prediction models achieve excellent performance (R² > 0.92), indicating strong predictability
- Tip prediction is more challenging (R² ≈ 0.56), reflecting the inherent variability in customer tipping behavior
- Both model types (Linear and Lasso) perform similarly, suggesting the dataset doesn't suffer from significant multicollinearity

## How to Use This Code

### Prerequisites

Ensure you have the following Python packages installed:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Running the Analysis

#### Option 1: Full Training Pipeline
If you want to retrain models from scratch:

1. **Run the training notebook**:
   ```bash
   jupyter notebook training.ipynb
   ```
   This will:
   - Load and preprocess the raw data
   - Perform exploratory data analysis
   - Train and tune both fare and tip prediction models
   - Save trained models and preprocessing pipelines

2. **Evaluate on test set**:
   ```bash
   jupyter notebook test.ipynb
   ```

#### Option 2: Using Pre-trained Models (Recommended)
For quick evaluation using existing trained models:

1. **Direct evaluation**:
   ```bash
   jupyter notebook test.ipynb
   ```
   This notebook loads pre-trained models and evaluates them on the test set without requiring any training.

### Making Predictions on New Data

To use the trained models for predictions on new taxi trip data:

```python
import joblib
import pandas as pd

# Load preprocessing pipeline and models
preprocessor = joblib.load('data/processed_data/preprocessing_pipeline.pkl')
fare_model = joblib.load('models/fare_prediction/lasso_regression.pkl')
tip_model = joblib.load('models/tip_prediction/lasso_regression.pkl')

# Load your new data
new_data = pd.read_csv('your_new_taxi_data.csv')

# Preprocess the data
processed_data = preprocessor.transform(new_data)

# Make predictions
fare_predictions = fare_model.predict(processed_data)
tip_predictions = tip_model.predict(processed_data)

print(f"Predicted fares: {fare_predictions}")
print(f"Predicted tips: {tip_predictions}")
```

## Business Applications

### For Taxi Companies
- **Dynamic pricing**: Use fare predictions to optimize pricing strategies
- **Route optimization**: Identify high-value routes and time periods
- **Driver incentives**: Use tip predictions to design driver bonus programs

### For Drivers
- **Shift planning**: Focus on high-tip locations and times
- **Customer service**: Understand factors that influence tipping
- **Earnings estimation**: Predict daily/weekly earnings potential

### For City Planning
- **Traffic management**: Understand taxi demand patterns
- **Infrastructure planning**: Identify high-traffic areas needing attention
- **Economic analysis**: Assess taxi industry contribution to local economy

## Technical Implementation Details

- **Framework**: scikit-learn for machine learning pipelines
- **Data processing**: Pandas for data manipulation and cleaning
- **Visualization**: Matplotlib and Seaborn for exploratory analysis
- **Model persistence**: Joblib for saving/loading trained models
- **Validation**: Cross-validation with confidence intervals for robust evaluation

## Future Enhancements

1. **Advanced models**: Implement ensemble methods (Random Forest, Gradient Boosting)
2. **Deep learning**: Explore neural networks for complex pattern recognition
3. **Real-time prediction**: Deploy models as web services for live predictions
4. **Geographic analysis**: Incorporate detailed location-based features
5. **Weather integration**: Add weather data to improve prediction accuracy

## Contact

For questions about this analysis or suggestions for improvements, please refer to the detailed implementation in the Jupyter notebooks or the accompanying technical report.
        