import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine

# Load the Wine dataset
wine = load_wine()
df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
df['target'] = wine.target

# Display basic information
print(df.head())
print(df.describe())
print(df.info())

# Descriptive statistics
mean = df.mean()
median = df.median()
mode = df.mode().iloc[0]
std_dev = df.std()
variance = df.var()

print("Mean:\n", mean)
print("Median:\n", median)
print("Mode:\n", mode)
print("Standard Deviation:\n", std_dev)
print("Variance:\n", variance)

# Histograms
df.hist(figsize=(10, 10))
plt.show()

# Pair plot
sns.pairplot(df, hue='target', markers=["o", "s", "D"])
plt.show()

# Box plots
df.plot(kind='box', subplots=True, layout=(6,3), figsize=(15, 20))
plt.show()

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.drop('target', axis=1))

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

# LDA
lda = LDA(n_components=2)
lda_result = lda.fit_transform(scaled_data, df['target'])
df['LDA1'] = lda_result[:, 0]
df['LDA2'] = lda_result[:, 1]

# Plot PCA and LDA results
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x='PCA1', y='PCA2', hue='target', data=df, palette='Set1')
plt.title('PCA of Wine dataset')

plt.subplot(1, 2, 2)
sns.scatterplot(x='LDA1', y='LDA2', hue='target', data=df, palette='Set1')
plt.title('LDA of Wine dataset')

plt.show()

# Linear Regression
X = df[wine.feature_names]
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predictions
y_pred = lr.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_data)

# Plot clustering results
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='cluster', data=df, palette='Set2')
plt.title('K-Means Clustering of Wine dataset')
plt.show()
