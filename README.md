# Recommendation-System---Books-
This project builds an end-to-end Book Recommendation System using users.csv, ratings.csv, and books.csv. Data is cleaned, merged, and analyzed using EDA. Collaborative filtering with Truncated SVD is applied to generate user-based and item-based recommendations, and the model is deployed using Streamlit.

Truncated SVD (Singular Value Decomposition) is a dimensionality reduction technique used to simplify large matrices while preserving the most important patterns.
In simple terms:
It breaks a large matrix (like a user–item rating matrix) into smaller matrices that capture the most meaningful relationships and ignores noise.

This project is an end-to-end Book Recommendation System built using user interaction data. The workflow starts with three raw datasets: users.csv, ratings.csv, and books.csv.

First, the datasets are cleaned and preprocessed, including handling missing values, filtering invalid records, and standardizing columns. The cleaned datasets are then merged to create a unified user–book interaction dataset.
Exploratory Data Analysis (EDA) is performed to understand user activity, rating distribution, popular books, and sparsity in the user–item matrix.

For model building, the merged data is transformed into a user–item rating matrix. Collaborative Filtering is implemented using Truncated SVD (Matrix Factorization) to learn latent representations of users and books. Recommendation scores are computed using dot product similarity, enabling both item-based and user-based recommendations.

For deployment, the trained model components and mappings are serialized using pickle. A Streamlit web application loads these artifacts to serve real-time recommendations, display book details, generate personalized suggestions, and provide basic user analytics through visualizations.

This project demonstrates a complete data science pipeline, from raw data processing and analysis to model development and deployment.


