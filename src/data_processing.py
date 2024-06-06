import pandas as pd
from scipy.stats import zscore

# Load the dataset
file_path = 'data/train/train.csv'
data = pd.read_csv(file_path)

# Calculate Z-scores
z_scores = data[['temp', 'atemp', 'hum', 'windspeed', 'cnt']].apply(zscore)

# Identify outliers (Z-score method: abs(z) > 3)
outliers_z = (z_scores.abs() > 3)

# IQR method
Q1 = data[['temp', 'atemp', 'hum', 'windspeed', 'cnt']].quantile(0.25)
Q3 = data[['temp', 'atemp', 'hum', 'windspeed', 'cnt']].quantile(0.75)
IQR = Q3 - Q1

# Identify outliers (IQR method: outside 1.5*IQR)
outliers_iqr = ((data[['temp', 'atemp', 'hum', 'windspeed', 'cnt']] < (Q1 - 1.5 * IQR)) | 
                (data[['temp', 'atemp', 'hum', 'windspeed', 'cnt']] > (Q3 + 1.5 * IQR)))

# Combine both methods
combined_outliers = outliers_z | outliers_iqr

# Correct outliers by replacing them with the median of the column
for column in ['temp', 'atemp', 'hum', 'windspeed', 'cnt']:
    median_value = data[column].median()
    data.loc[combined_outliers[column], column] = median_value

# Save the cleaned dataset
cleaned_file_path = 'data/train/cleaned_train.csv'
data.to_csv(cleaned_file_path, index=False)

print("Outliers have been corrected and the cleaned dataset has been saved.")
