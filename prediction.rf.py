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
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
import matplotlib.ticker as mticker

csv_url_data = "https://raw.githubusercontent.com/Eamon-Sinclair/PitcherContract/main/Pitcher_Data.csv"
csv_url_contract = "https://raw.githubusercontent.com/Eamon-Sinclair/PitcherContract/refs/heads/main/FAContract.csv"

variables = ["WARP", "Age", "DRA", "GS", "IPGS", "ERA", "RA9", "FIP", "WHIP", "K",
             "BB", "KBB", "Whiff", "Swing", "OSwing", "ZSwing", "OContact", "ZContact",
             "Contact", "Zone", "CSProb", "CStr"]

# Function to load data
def load_data(file_url):
    data = pd.read_csv(file_url)
    data = data.dropna()
    return data

Data = load_data(csv_url_data)
aav_data = load_data(csv_url_contract)

#Remove Accents
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

#Functions
def cluster_players(Combined, variables, n_clusters):
    numeric_columns = Combined[variables]
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(numeric_columns)
    kmeans = KMeans(n_clusters=n_clusters, random_state=123)
    clusters = kmeans.fit_predict(normalized_data)
    Combined['Cluster'] = clusters
    return Combined, kmeans, scaler

Combined, kmeans_model, scaler = cluster_players(Combined, variables, n_clusters=4)

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

def get_important_features_for_cluster(predicted_cluster, cluster_models, top_n=7):
    model = cluster_models.get(predicted_cluster)
    
    if model and hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'Variable': variables,
            'Importance': feature_importance
        })
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        
        return importance_df.head(top_n)['Variable'].tolist()
    else:
        return []

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

def predict_cluster_for_new_player(player_data, kmeans_model, scaler, aav_variables):
    scaled_data = scaler.transform(player_data[aav_variables])
    predicted_cluster = kmeans_model.predict(scaled_data)
    return predicted_cluster[0]

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

st.title("Starting Pitcher Contract Model")

st.markdown("""
    **Created by Eamon Sinclair and Matthew Krakower**  
    **Twitter**: [EamonSinclair](https://x.com/_EamonSinclair), [MatthewKrakower](https://x.com/MatthewKrakower)  
    **Email**: [eamonsinclair15@gmail.com](mailto:eamonsinclair15@gmail.com), [Matthew.krakower21@gmail.com](mailto:Matthew.krakower21@gmail.com)    
    **Linkedin**: [EamonSinclair](https://www.linkedin.com/in/eamonsinclair/), [MatthewKrakower](https://www.linkedin.com/in/matthew-krakower-8b657827b/)
""", unsafe_allow_html=True)

#Process Data
player_name = st.selectbox("Select a Starting Pitcher", Data['Player'].unique())
player_year = 2024

#Prediction Table
new_player_data = Data[(Data['Player'] == player_name) & (Data['Year'] == player_year)]

if new_player_data.empty:
    st.error("Pitcher Did Not Pitch in Selected Year")
else:
    new_player_data = new_player_data[variables]

    with st.spinner("Simulating predictions..."):
        predicted_aav_values, median_aav, aav_confidence_interval, predicted_length_values, median_length, length_confidence_interval = simulate_player_season(
            new_player_data, kmeans_model, scaler, variables, length_model, variables, n_simulations=250, max_deviation=0.03
        )

    predicted_total = median_aav * round(median_length)

    predicted_values = {
        "Contract": ["AAV", "Length", "Total Value"],
        "Lower Bound Estimate": [
            f"${aav_confidence_interval[0]:,.2f}",
            f"{length_confidence_interval[0]:,.2f} years",
            f"${aav_confidence_interval[0] * round(length_confidence_interval[0]):,.2f}" 
        ],
        "Best Estimate": [
            f"${median_aav:,.2f}",
            f"{median_length:.2f} years",
            f"${median_aav * round(median_length):,.2f}" 
        ],
        "Upper Bound Estimate": [
            f"${aav_confidence_interval[1]:,.2f}",
            f"{length_confidence_interval[1]:,.2f} years",
            f"${aav_confidence_interval[1] * round(length_confidence_interval[1]):,.2f}" 
        ]
    }

    recent_contract = Combined[Combined['Player'] == player_name].sort_values(by='Year', ascending=False).head(1)

    if not recent_contract.empty:
        recent_aav = recent_contract.iloc[0]['AAV']
        recent_length = recent_contract.iloc[0]['Length']
        recent_total = recent_aav * recent_length

        predicted_values["Actual"] = [
            f"${recent_aav:,.2f}",
            f"{recent_length:.2f} years",
            f"${recent_total:,.2f}"
        ]
    else:
        predicted_values["Actual"] = ["No FA Contract From '21 - '24", "N/A", "N/A"]

    # Create the DataFrame for the table
    table_df = pd.DataFrame(predicted_values)

    # Convert the table to HTML and apply CSS to center the column headers
    html_table = table_df.to_html(index=False, escape=False)

    # Custom CSS to center the column headers
    css = """
    <style>
        th {
            text-align: center !important;
        }
        td {
            text-align: center !important;
        }
        .dataframe th:nth-child(3), .dataframe td:nth-child(3) {  /* Targets the "Best Estimate" column */
            background-color: #D3D3D3 !important;
        }
    </style>
    """

    # Display the styled table
    st.markdown(css, unsafe_allow_html=True)
    st.write(html_table, unsafe_allow_html=True)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # AAV Histogram
    kde_aav = gaussian_kde(predicted_aav_values)

    aav_min = min(predicted_aav_values)
    aav_max = max(predicted_aav_values)
    aav_range = np.linspace(aav_min - 0.05 * (aav_max - aav_min), 
                            aav_max + 0.05 * (aav_max - aav_min), 500)
    kde_aav_values = kde_aav(aav_range)

    axs[0].plot(aav_range, kde_aav_values, color='skyblue', label='_nolegend_')
    axs[0].fill_between(aav_range, kde_aav_values, color='skyblue', alpha=0.3)

    axs[0].axvline(aav_confidence_interval[0], color='red', linestyle='--', label='Lower Bound Esimate')
    axs[0].axvline(aav_confidence_interval[1], color='red', linestyle='--', label='Upper Bound Estimate')
    axs[0].axvline(median_aav, color='green', linestyle='-', label='Best Estimate')

    axs[0].set_title(f"{player_name}: Simulated AAV Predictions")
    axs[0].set_xlabel("AAV ($ millions)")
    axs[0].set_ylabel("Density")
    axs[0].legend()

    axs[0].yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    axs[0].yaxis.get_offset_text().set_visible(False)  # Hide offset text if present
    axs[0].ticklabel_format(style='plain', axis='y')

    y_ticks = axs[0].get_yticks()
    axs[0].set_yticklabels([f'{round(tick * 1e7, 1):,}' for tick in y_ticks])

    ticks = axs[0].get_xticks()
    axs[0].set_xticklabels([f'{int(tick / 1_000_000)}' for tick in ticks])

    #Length Histogram

    kde_length = gaussian_kde(predicted_length_values)
    length_min = min(predicted_length_values)
    length_max = max(predicted_length_values)
    length_range = np.linspace(length_min - 0.05 * (length_max - length_min), 
                               length_max + 0.05 * (length_max - length_min), 500)
    kde_length_values = kde_length(length_range)
    axs[1].plot(length_range, kde_length_values, color='lightgreen', label='_nolegend_')
    axs[1].fill_between(length_range, kde_length_values, color='lightgreen', alpha=0.3)
    axs[1].axvline(length_confidence_interval[0], color='red', linestyle='--', label='Lower Bound Estimate')
    axs[1].axvline(length_confidence_interval[1], color='red', linestyle='--', label='Upper Bound Estimate')
    axs[1].axvline(median_length, color='green', linestyle='-', label='Best Estimate')
    axs[1].set_title(f"{player_name}: Simulated Length Predictions")
    axs[1].set_xlabel("Length (years)")
    axs[1].set_ylabel("Density")
    axs[1].legend()

    st.pyplot(fig)

#Similarity Table
def get_most_similar_players(player_stats, cluster_players_stats, important_stats, Combined):
    distances = euclidean_distances(player_stats[important_stats], cluster_players_stats[important_stats])

    sorted_indices = distances.argsort(axis=1)[:, :5]
    most_similar_players = cluster_players_stats.iloc[sorted_indices.flatten()]

    most_similar_players = pd.concat([Combined[['Player', 'Year', 'AAV','Length']], most_similar_players], axis=1)
    
    most_similar_players = most_similar_players[['Player', 'Year'] + important_stats + ['AAV', 'Length']]
    most_similar_players['Year'] = most_similar_players['Year'].astype(str)
    
    most_similar_players = most_similar_players.dropna()

    most_similar_players = most_similar_players.set_index('Player')
    
    return most_similar_players

def display_prediction_table(player_name, player_year, predicted_aav, predicted_length, predicted_cluster,
                             Combined, variables, aav_models, length_model,
                             important_stats):

    median_aav = predicted_aav 
    median_length = predicted_length 

    if isinstance(predicted_aav, (list, tuple)):
        median_aav = predicted_aav[0]
    
    if isinstance(predicted_length, (list, tuple)):
        median_length = predicted_length[0]

    important_stats = get_important_features_for_cluster(predicted_cluster, aav_models)

    player_stats = Data[(Data['Player'] == player_name) & (Data['Year'] == player_year)]
    if not player_stats.empty:
        player_stats = player_stats[important_stats]
        player_stats = player_stats.transpose()
        player_stats.columns = [f"{player_name} in {player_year}"]
        player_stats = player_stats.T
    else:
        player_stats = pd.DataFrame(columns=[f"{player_name} in {player_year}"])

    player_stats['Model AAV'] = median_aav
    player_stats['Model Length'] = round(median_length)

    st.write(f"### {player_name}'s 2024 Season Stats")
    st.dataframe(player_stats)

    cluster_players_stats = Combined[Combined['Cluster'] == predicted_cluster]
    cluster_players_stats = cluster_players_stats[cluster_players_stats['Player'] != player_name]

    if not cluster_players_stats.empty:
        st.write("### Most Similar Pitcher Seasons Between 2021 - 2024:")

        most_similar_players = get_most_similar_players(
            player_stats[important_stats], cluster_players_stats[important_stats], important_stats, Combined
        )

        st.dataframe(most_similar_players)
    else:
        st.write("### Most Similar Players in Same Cluster: Not found")


predicted_cluster = predict_cluster_for_new_player(new_player_data, kmeans_model, scaler, variables)
important_stats = get_important_features_for_cluster(predicted_cluster, aav_models)

display_prediction_table(
    player_name=player_name,
    player_year=player_year,
    predicted_aav=median_aav,  
    predicted_length=median_length,  
    predicted_cluster=predicted_cluster,
    Combined=Combined,  
    variables=variables, 
    aav_models=aav_models,  
    length_model=length_model,  
    important_stats=important_stats  
)
