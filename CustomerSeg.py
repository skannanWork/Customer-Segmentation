import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the customer data into a Pandas DataFrame
df = pd.read_csv('customer_data.csv')

# Select the features and target column
X = df[['age', 'income', 'spend']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create the segments using K-Means clustering
kmeans = KMeans(n_clusters=4)
kmeans.fit(X_scaled)

# Assign the segments to the customers
df['segment'] = kmeans.labels_

# Print the resulting segments
print(df)
