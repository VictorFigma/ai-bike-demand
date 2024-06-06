import json
import joblib
import pandas as pd
from scipy.stats import zscore

# Load the saved model
model_file_path = 'models/model.pkl'
model = joblib.load(model_file_path)
print(f'Model loaded from {model_file_path}')

# Load the dataset or new data for prediction
predict_data_file_path = 'data/test/test.csv'
predict_data = pd.read_csv(predict_data_file_path)

# Calculate Z-scores
z_scores = predict_data[['temp', 'atemp', 'hum', 'windspeed']].apply(zscore)

# Identify outliers (Z-score method: abs(z) > 3)
outliers_z = (z_scores.abs() > 3)

# IQR method
Q1 = predict_data[['temp', 'atemp', 'hum', 'windspeed']].quantile(0.25)
Q3 = predict_data[['temp', 'atemp', 'hum', 'windspeed']].quantile(0.75)
IQR = Q3 - Q1

# Identify outliers (IQR method: outside 1.5*IQR)
outliers_iqr = ((predict_data[['temp', 'atemp', 'hum', 'windspeed']] < (Q1 - 1.5 * IQR)) | 
                (predict_data[['temp', 'atemp', 'hum', 'windspeed']] > (Q3 + 1.5 * IQR)))

# Combine both methods
combined_outliers = outliers_z | outliers_iqr

# Correct outliers by replacing them with the median of the column
for column in ['temp', 'atemp', 'hum', 'windspeed']:
    median_value = predict_data[column].median()
    predict_data.loc[combined_outliers[column], column] = median_value

# Prepare features
X_predict = predict_data[['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']]

# Make predictions
predictions = model.predict(X_predict)

# Add predictions to the dataset
predict_data['predicted_cnt'] = predictions

# Create a dictionary for the required format
predictions_dict = {
    "target": {
        row['dteday'] + f" {int(row['hr']):02}:00": int(pred)
        for _, row in predict_data.iterrows()
        for pred in [row['predicted_cnt']]
    }
}

# Save the predictions to a JSON file
predictions_file_path = 'predictions/predictions.json'
with open(predictions_file_path, 'w') as f:
    json.dump(predictions_dict, f, indent=4)

print(f'Predictions saved to {predictions_file_path}')