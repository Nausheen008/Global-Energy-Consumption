# Global-Energy-Consumption and CO2 emission Analysis 
This repo contains the Energy consumption of diff. countries along with their CO2 emission 
This project analyzes 22k+ global energy records across 184 countries (2000–2022) to understand energy consumption patterns and their relationship with CO₂ emissions.

It integrates Exploratory Data Analysis (EDA), Clustering, and Machine Learning into a single workflow to uncover insights and build predictive models for policy and decision-making.

🔬 Key Objectives

Analyze global energy consumption trends and emissions drivers

Segment countries using K-Means clustering based on energy mix & economic indicators

Build and validate a Random Forest regression model to predict CO₂ per capita

Identify key predictors of emissions (e.g., electricity per capita, GDP, fossil fuel share)

Generate interactive dashboards & visualizations for stakeholders

📂 Dataset

Source: Synthetic yet realistic dataset (global_energy_dataset.csv) with 22k+ records

Scope: 184 countries × 23 years (2000–2022)

Features: Population, GDP, primary energy use, fossil/renewable/nuclear mix, electricity, CO₂ emissions

🛠️ Methodology

Exploratory Data Analysis (EDA)

Global CO₂ emission trends

Renewable vs fossil share evolution

Country-level emission comparisons

Clustering (K-Means)

Segmented countries into clusters based on GDP, CO₂, and energy mix

Optimal cluster number determined via silhouette score

Machine Learning (Random Forest Regression)

Target: CO₂ emissions per capita

Achieved R² = 0.89 on test set (target = 0.82)

Identified top features driving emissions

Validation

5-Fold Cross-Validation

Feature importance analysis

Stability and error analysis

Visualization & Dashboards

Global CO₂ emission trends

GDP vs CO₂ relationship

Energy mix evolution

Feature importance & model performance plots

📊 Results & Insights

Countries segmented into distinct energy-economic clusters

Random Forest Model achieved R² = 0.89, exceeding target

Top drivers of CO₂ per capita:

Electricity per capita

Primary energy consumption

Coal share in energy mix

📁 Generated Outputs

global_energy_dataset_final.csv – Final processed dataset

country_energy_clusters.csv – Cluster assignments per country

feature_importance_analysis.csv – Feature importance scores

comprehensive_eda_dashboard.png – Global trends dashboard

kmeans_clustering_results.png – Clustering visualization

random_forest_performance.png – Model performance plots

⚙️ Tech Stack

Languages: Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)

ML Models: K-Means, Random Forest Regression

Validation: Cross-validation, Feature Importance, R², RMSE

Visualization: Matplotlib, Seaborn

🚀 How to Run
# Clone the repo
git clone https://github.com/<your-username>/Global-Energy-Consumption.git
cd Global-Energy-Consumption

# Install dependencies
pip install -r requirements.txt

# Run the main script
python comprehensive_energy_analysis_complete.py

🎯 Future Improvements

Incorporate real-world datasets (IEA, World Bank, Our World in Data)

Add interactive dashboards (Dash/Streamlit)

Explore time-series forecasting for future CO₂ emissions

Extend ML models with Gradient Boosting/XGBoost

✍️ Author: Nausheen Farhat
📌 IIT Delhi | Data Science & Analytics Enthusiast
