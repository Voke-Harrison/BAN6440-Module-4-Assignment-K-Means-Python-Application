# BAN6440 - Applied Machine Learning for Analytics
# Module 4 Assignment: K-Means Python Application
# Voke Harrison Edafejimue
# Learner ID - 143304
# Purpose: This application uses the K-Means clustering algorithm to group hospitals based on the COVID-19 severity.
# By analyzing the dataset, the objective is to identify patterns and make data-driven insights to improve healthcare strategies.

# import necessary libraries

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)


# Load dataset (CSV downloaded from AWS)
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        logging.error("File not found. Please check the file path.")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


# Preprocess data
def preprocess_data(df):
    try:
        # Extract relevant columns (7-day severity and hospital-related columns)
        severity_columns = ['severity_1-day', 'severity_2-day', 'severity_3-day', 'severity_4-day', 'severity_5-day',
                            'severity_6-day', 'severity_7-day']
        data_for_clustering = df[severity_columns].copy()

        # Handle missing data (if any)
        data_for_clustering = data_for_clustering.fillna(data_for_clustering.mean())
        logging.info("Data preprocessing successful. Missing values filled with mean.")

        return data_for_clustering
    except KeyError as e:
        logging.error(f"Column not found: {e}")
        raise
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise


# Standardize the data
def standardize_data(data):
    try:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        logging.info("Data standardized successfully.")
        return data_scaled
    except Exception as e:
        logging.error(f"Error during standardization: {e}")
        raise


# Perform K-Means clustering
def perform_kmeans(data, n_clusters=3):
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data)
        logging.info(f"K-Means clustering completed with {n_clusters} clusters.")
        return kmeans
    except Exception as e:
        logging.error(f"Error during K-Means clustering: {e}")
        raise


# Elbow Method for determining optimal clusters
def elbow_method(data):
    try:
        inertia = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(data)
            inertia.append(kmeans.inertia_)

        plt.plot(range(1, 11), inertia, marker='o')
        plt.title("Elbow Method For Optimal k")
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.show()

    except Exception as e:
        logging.error(f"Error in Elbow method: {e}")
        raise


# Perform PCA for dimensionality reduction and visualization
def perform_pca(data):
    try:
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(data)
        logging.info("PCA performed successfully.")
        return pca_components
    except Exception as e:
        logging.error(f"Error during PCA: {e}")
        raise


# Visualize clusters in PCA space
def visualize_clusters_pca(df):
    try:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=100)
        plt.title('PCA of K-Means Clusters')
        plt.show()
        logging.info("Cluster visualization completed.")
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        raise


# Unit testing for loading data
def test_load_data():
    try:
        df = load_data('severity-index.csv')
        assert df is not None, "Data loading failed: DataFrame is empty."
        logging.info("test_load_data passed")
    except AssertionError as e:
        logging.error(f"test_load_data failed: {e}")
        raise


# Unit testing for data preprocessing
def test_preprocess_data():
    try:
        df = load_data('severity-index.csv')
        processed_data = preprocess_data(df)
        assert processed_data.isnull().sum().sum() == 0, "Missing values found after preprocessing."
        logging.info("test_preprocess_data passed")
    except AssertionError as e:
        logging.error(f"test_preprocess_data failed: {e}")
        raise


# Unit testing for clustering
def test_kmeans_clustering():
    try:
        # Use simple mock data for testing
        test_data = np.array([[1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7, 8, 9]])
        kmeans_test = perform_kmeans(test_data)
        assert len(
            kmeans_test.labels_) == 3, "Test failed: The number of labels should match the number of data points."
        logging.info("test_kmeans_clustering passed")
    except AssertionError as e:
        logging.error(f"test_kmeans_clustering failed: {e}")
        raise


# Unit testing for PCA
def test_pca():
    try:
        # Use simple mock data for PCA
        test_data = np.array([[1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7, 8, 9]])
        pca_components = perform_pca(test_data)
        assert pca_components.shape[1] == 2, "Test failed: PCA should output 2 components."
        logging.info("test_pca passed")
    except AssertionError as e:
        logging.error(f"test_pca failed: {e}")
        raise


# Main execution
def main():
    try:
        # Load data
        df = load_data('severity-index.csv')

        # Preprocess the data
        processed_data = preprocess_data(df)

        # Standardize data
        standardized_data = standardize_data(processed_data)

        # Apply KMeans clustering
        elbow_method(standardized_data)  # Determine optimal number of clusters
        kmeans_model = perform_kmeans(standardized_data, n_clusters=3)
        df['Cluster'] = kmeans_model.labels_

        # Apply PCA
        pca_components = perform_pca(standardized_data)
        df['PCA1'] = pca_components[:, 0]
        df['PCA2'] = pca_components[:, 1]

        # Visualize results
        visualize_clusters_pca(df)

        logging.info("Application executed successfully.")
    except Exception as e:
        logging.error(f"Error during execution: {e}")


# Unit tests
def run_unit_tests():
    test_load_data()
    test_preprocess_data()
    test_kmeans_clustering()
    test_pca()


# Run the application and tests
if __name__ == "__main__":
    run_unit_tests()
    main()
