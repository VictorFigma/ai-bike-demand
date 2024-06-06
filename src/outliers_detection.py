from scipy.stats import zscore
import pandas as pd

#file_path = 'data/train/cleaned_train.csv'
file_path = 'data/train/train.csv'
data = pd.read_csv(file_path)

# Calculate Z-scores
z_scores = data[['temp', 'atemp', 'hum', 'windspeed', 'cnt']].apply(zscore)

# Identify outliers (Z-score method: abs(z) > 3)
outliers_z = (z_scores.abs() > 3).any(axis=1)

# IQR method
Q1 = data[['temp', 'atemp', 'hum', 'windspeed', 'cnt']].quantile(0.25)
Q3 = data[['temp', 'atemp', 'hum', 'windspeed', 'cnt']].quantile(0.75)
IQR = Q3 - Q1

# Identify outliers (IQR method: outside 1.5*IQR)
outliers_iqr = ((data[['temp', 'atemp', 'hum', 'windspeed', 'cnt']] < (Q1 - 1.5 * IQR)) | 
                (data[['temp', 'atemp', 'hum', 'windspeed', 'cnt']] > (Q3 + 1.5 * IQR))).any(axis=1)

# Summarize the results
outliers_summary = {
    'Total rows': len(data),
    'Outliers (Z-score method)': outliers_z.sum(),
    'Outliers (IQR method)': outliers_iqr.sum(),
    'Common outliers': (outliers_z & outliers_iqr).sum()
}

print(outliers_summary)