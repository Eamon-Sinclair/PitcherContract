import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import unicodedata
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score

# URL for the CSV files hosted on GitHub
csv_url_data = "https://raw.githubusercontent.com/Eamon-Sinclair/PitcherContract/main/Pitcher_Data.csv"
csv_url_contract = "https://raw.githubusercontent.com/Eamon-Sinclair/PitcherContract/refs/heads/main/FAContract.csv"

# Variables for modeling
variables = ["WARP", "Age", "DRA", "GS", "IPGS", "ERA", "RA9", "FIP", "WHIP", "K",
             "BB", "KBB", "Whiff", "Swing", "OSwing", "ZSwing", "OContact", "ZContact",
             "Contact", "Zone", "CSProb", "CStr"]

# Function to load data
def load_data(file_url):
    data = pd.read_csv(file_url)
    data = data.dropna()
    return data

# Load data from GitHub CSV URLs
Data = load_data(csv_url_data)
aav_data = load_data(csv_url_contract)

# Function to remove accents from player names
def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

Data['Player'] = Data['Player'].apply(remove_accents)
aav_data['Player'] = aav_data['Player'].apply(remove_accents)
aav_data["YearMinusOne"] = aav_data["Year"] - 1

# Merge datasets
Combined = pd.merge(Data, aav_data, left_on=['Player', 'Year'], right_on=['Player', 'YearMinusOne'], how='left')
Combined = Combined.dropna(subset=['AAV'])
Combined = Combined.drop(columns=['Year_y', 'YearMinusOne'])
Combined = Combined.rename(columns={'Year_x': 'Year'})

# Clustering function
def cluster_players(Combined, variables, n_clusters):
    numeric_columns = Combined[variables]
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(numeric_columns)
    kmeans = KMeans(n_clusters=n_clusters, random_state=123)
    clusters = kmeans.fit_predict(normalized_data)
    Combined['Cluster'] = clusters
    return Combined, kmeans, scaler

Combined, kmeans_model, scaler = cluster_players(Combined, variables, n_clusters=4)

# XGBoost model training for AAV
def train_xgb_for_clusters(Combined, variables, target_column):
    cluster_models = {}
    for cluster_id in Combined['Cluster'].unique():
        cluster_data = Combined[Combined['Cluster'] == cluster_id]
        X = cluster_data[variables]
        y = cluster_data[target_column]
        xgb_model = xgb.XGBRegressor(
            random_state=123,
            max_depth=4,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=10,
            alpha=5,
            learning_rate=0.05,
            n_estimators=500
        )
        xgb_model.fit(X, y)
        cluster_models[cluster_id] = xgb_model
    return cluster_models

aav_models = train_xgb_for_clusters(Combined, variables, target_column="AAV")

# XGBoost model training for Length
def length_xgb(Combined, variables, target_column):
    X = Combined[variables]
    y = Combined[target_column]
    xgb_model = xgb.XGBRegressor(
        random_state=123,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=10,
        alpha=5,
        learning_rate=0.05,
        n_estimators=500
    )
    xgb_model.fit(X, y)
    return xgb_model

length_model = length_xgb(Combined, variables, target_column="Length")

# Function to predict cluster for new player
def predict_cluster_for_new_player(player_data, kmeans_model, scaler, aav_variables):
    scaled_data = scaler.transform(player_data[aav_variables])
    predicted_cluster = kmeans_model.predict(scaled_data)
    return predicted_cluster[0]

# Simulate season and predict contract values
def simulate_player_season(player_data, kmeans_model, scaler, aav_variables, length_model, variables, n_simulations=250, max_deviation=0.03):
    predicted_aav_values = []
    predicted_length_values = []

    for _ in range(n_simulations):
        deviation_factor = 1 + np.random.uniform(-max_deviation, max_deviation)
        simulated_data = player_data.copy()
        simulated_data *= deviation_factor

        predicted_cluster = predict_cluster_for_new_player(simulated_data, kmeans_model, scaler, aav_variables)
        aav_model = aav_models[predicted_cluster]
        simulated_aav = aav_model.predict(simulated_data[aav_variables])[0]
        predicted_aav_values.append(simulated_aav)

        simulated_length = length_model.predict(simulated_data[variables])[0]
        predicted_length_values.append(simulated_length)

    aav_median = np.median(predicted_aav_values)
    aav_confidence_interval = (np.percentile(predicted_aav_values, 2.5), np.percentile(predicted_aav_values, 97.5))

    length_median = np.median(predicted_length_values)
    length_confidence_interval = (np.percentile(predicted_length_values, 2.5), np.percentile(predicted_length_values, 97.5))

    return predicted_aav_values, aav_median, aav_confidence_interval, predicted_length_values, length_median, length_confidence_interval

# Streamlit Interface
st.title("Pitcher Contract Prediction Tool")

# Preprocess data
player_name = st.selectbox("Select a player", Data['Player'].unique())
player_year = st.number_input("Free Agency Year", min_value=2021, max_value=2024, value=2024)

# Check if player exists and pitched that year
new_player_data = Data[(Data['Player'] == player_name) & (Data['Year'] == player_year)]

if new_player_data.empty:
    st.error("Pitcher Did Not Pitch in Selected Year")
else:
    new_player_data = new_player_data[variables]

    # Simulate predictions
    with st.spinner("Simulating predictions..."):
        predicted_aav_values, median_aav, aav_confidence_interval, predicted_length_values, median_length, length_confidence_interval = simulate_player_season(
            new_player_data, kmeans_model, scaler, variables, length_model, variables, n_simulations=250, max_deviation=0.03
        )

    predicted_total = median_aav * round(median_length)

    # Display predictions
    st.write(f"Predicted AAV: ${median_aav:,.2f}")
    st.write(f"Predicted Length: {median_length:.2f} years")
    st.write(f"Predicted Total Contract Value: ${predicted_total:,.2f}")

    # Plot histograms for both predictions
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # AAV Histogram
    kde_aav = gaussian_kde(predicted_aav_values)

    aav_min = min(predicted_aav_values)
    aav_max = max(predicted_aav_values)
    aav_range = np.linspace(aav_min - 0.05 * (aav_max - aav_min), 
                            aav_max + 0.05 * (aav_max - aav_min), 500)
    kde_aav_values = kde_aav(aav_range)

    # Plot the KDE
    axs[0].plot(aav_range, kde_aav_values, color='skyblue', label='KDE')
    axs[0].fill_between(aav_range, kde_aav_values, color='skyblue', alpha=0.3)

    # Add confidence intervals
    axs[0].axvline(aav_confidence_interval[0], color='red', linestyle='--', label='Lower 95% CI')
    axs[0].axvline(aav_confidence_interval[1], color='red', linestyle='--', label='Upper 95% CI')
    axs[0].axvline(median_aav, color='green', linestyle='-', label='Median Prediction')

    # Title and labels
    axs[0].set_title(f"{player_name}: Simulated AAV Predictions (Smoothed)")
    axs[0].set_xlabel("AAV ($ millions)")  # Label the axis as millions
    axs[0].set_ylabel("Density")
    axs[0].legend()

    ticks = axs[0].get_xticks()
    axs[0].set_xticklabels([f'{int(tick / 1_000_000)}' for tick in ticks])  # Divide the tick by 1,000,000



    # Length Histogram
    kde_length = gaussian_kde(predicted_length_values)
    length_min = min(predicted_length_values)
    length_max = max(predicted_length_values)
    length_range = np.linspace(length_min - 0.05 * (length_max - length_min), 
                               length_max + 0.05 * (length_max - length_min), 500)
    kde_length_values = kde_length(length_range)
    axs[1].plot(length_range, kde_length_values, color='lightgreen', label='KDE')
    axs[1].fill_between(length_range, kde_length_values, color='lightgreen', alpha=0.3)
    axs[1].axvline(length_confidence_interval[0], color='red', linestyle='--', label='Lower 95% CI')
    axs[1].axvline(length_confidence_interval[1], color='red', linestyle='--', label='Upper 95% CI')
    axs[1].axvline(median_length, color='green', linestyle='-', label='Median Prediction')
    axs[1].set_title(f"{player_name}: Simulated Length Predictions (Smoothed)")
    axs[1].set_xlabel("Length (years)")
    axs[1].set_ylabel("Density")
    axs[1].legend()

    # Show the histograms
    st.pyplot(fig)
