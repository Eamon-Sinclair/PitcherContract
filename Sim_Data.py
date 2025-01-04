import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the necessary datasets
simulated_data = pd.read_csv("https://raw.githubusercontent.com/Eamon-Sinclair/PitcherContract/main/Sim_Data.csv")
contract_data = pd.read_csv("https://raw.githubusercontent.com/Eamon-Sinclair/PitcherContract/main/FAContract.csv")
similarity_data = pd.read_csv("https://raw.githubusercontent.com/Eamon-Sinclair/PitcherContract/main/Similar_Players.csv")
pitcher_data = pd.read_csv("https://raw.githubusercontent.com/Eamon-Sinclair/PitcherContract/main/Pitcher_Data.csv")
combined_data = pd.read_csv("https://raw.githubusercontent.com/Eamon-Sinclair/PitcherContract/main/Combined.csv")

# Page setup
st.title("Starting Pitcher Contract Model")

st.markdown("""
    **Created by Eamon Sinclair and Matthew Krakower**  
    **Twitter**: [EamonSinclair](https://x.com/_EamonSinclair), [MatthewKrakower](https://x.com/MatthewKrakower)  
    **Email**: [eamonsinclair15@gmail.com](mailto:eamonsinclair15@gmail.com), [Matthew.krakower21@gmail.com](mailto:Matthew.krakower21@gmail.com)    
    **Linkedin**: [EamonSinclair](https://www.linkedin.com/in/eamonsinclair/), [MatthewKrakower](https://www.linkedin.com/in/matthew-krakower-8b657827b/)         
    **Data**: [Baseball Prospectus](https://www.baseballprospectus.com/)
""", unsafe_allow_html=True)

# Player selection
players = simulated_data['Player'].unique()
player_name = st.selectbox("Select a Player", players)

if player_name:
    # Filter data for the selected player
    player_data = simulated_data[simulated_data['Player'] == player_name]
    
    # Extract bounds and best estimates
    lower_aav = player_data['Lower Bound AAV'].iloc[0]
    median_aav = player_data['Median AAV'].iloc[0]
    upper_aav = player_data['Upper Bound AAV'].iloc[0]
    lower_length = player_data['Lower Bound Length'].iloc[0]
    median_length = player_data['Median Length'].iloc[0]
    upper_length = player_data['Upper Bound Length'].iloc[0]

    # Calculate total contract values
    lower_total = lower_aav * round(lower_length)
    median_total = median_aav * round(median_length)
    upper_total = upper_aav * round(upper_length)

    # Find the most recent FA contract
    recent_contract = contract_data[contract_data['Player'] == player_name].sort_values(by='Year', ascending=False).head(1)
    if not recent_contract.empty:
        recent_aav = recent_contract.iloc[0]['AAV']
        recent_length = recent_contract.iloc[0]['Length']
        recent_total = recent_aav * recent_length
        last_contract = [f"${recent_aav:,.2f}", f"{recent_length:.2f} years", f"${recent_total:,.2f}"]
    else:
        last_contract = ["No FA Contract From '21 - '24", "N/A", "N/A"]

    # Prepare table data
    predicted_values = {
        "Contract": ["AAV", "Length", "Total Value"],
        "Lower Bound Estimate": [
            f"${lower_aav:,.2f}",
            f"{lower_length:.2f} years",
            f"${lower_total:,.2f}"
        ],
        "Best Estimate": [
            f"${median_aav:,.2f}",
            f"{median_length:.2f} years",
            f"${median_total:,.2f}"
        ],
        "Upper Bound Estimate": [
            f"${upper_aav:,.2f}",
            f"{upper_length:.2f} years",
            f"${upper_total:,.2f}"
        ],
        "Last FA Contract Signed": last_contract
    }
    table_df = pd.DataFrame(predicted_values)

    # Define custom CSS for shading the first table
    css = """
    <style>
        .predicted-table th {
            text-align: center !important;
        }
        .predicted-table td {
            text-align: center !important;
        }
        .predicted-table .blue-shade th:nth-child(3), .predicted-table .blue-shade td:nth-child(3) {
            background-color: #B9D9EB !important;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    # Prepare the predicted values table with the custom CSS applied
    predicted_values_html = table_df.to_html(index=False, escape=False)
    predicted_values_html = f'<div class="predicted-table blue-shade">{predicted_values_html}</div>'

    # Display the predicted values table with shading
    st.markdown(predicted_values_html, unsafe_allow_html=True)

    # Safely extract simulation data
    sim_aav = player_data[[col for col in simulated_data.columns if col.startswith("SimAAV")]].values.flatten()
    sim_length = player_data[[col for col in simulated_data.columns if col.startswith("SimLength")]].values.flatten()

    # Create subplots to display both graphs side by side
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))  # 1 row, 2 columns

    # Plot AAV Probability Density
    sns.kdeplot(sim_aav, fill=True, color="lightblue", label="AAV Distribution", ax=axes[0])
    axes[0].axvline(player_data['Median AAV'].iloc[0], color='red', linestyle='--', label='Median AAV')
    axes[0].axvline(player_data['Lower Bound AAV'].iloc[0], color='black', linestyle='--', label='Lower Bound AAV')  # Lower bound
    axes[0].axvline(player_data['Upper Bound AAV'].iloc[0], color='black', linestyle='--', label='Upper Bound AAV')  # Upper bound
    axes[0].set_xlabel("AAV ($M)")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"AAV Distribution for {player_name}")
    axes[0].legend(loc='upper right')

    # Plot Length Probability Density
    sns.kdeplot(sim_length, fill=True, color="lightgreen", label="Length Distribution", ax=axes[1])
    axes[1].axvline(player_data['Median Length'].iloc[0], color='red', linestyle='--', label='Median Length')
    axes[1].axvline(player_data['Lower Bound Length'].iloc[0], color='black', linestyle='--', label='Lower Bound Length')  # Lower bound
    axes[1].axvline(player_data['Upper Bound Length'].iloc[0], color='black', linestyle='--', label='Upper Bound Length')  # Upper bound
    axes[1].set_xlabel("Contract Length (Years)")
    axes[1].set_ylabel("Density")
    axes[1].set_title(f"Length Distribution for {player_name}")
    axes[1].legend(loc='upper right')

    # Display the plots
    st.pyplot(fig)

    # Define cluster-specific stats
    cluster_stats = {
        0: ['WARP', 'IPGS', 'GS', 'DRA', 'CSProb', 'Age'],
        1: ['IPGS', 'WHIP', 'Swing', 'RA9', 'DRA', 'Age'],
        2: ['DRA', 'KBB', 'WARP', 'Age', 'ZContact', 'CSProb']
    }

    # Fetch target player's cluster from Combined data
    target_player_cluster = simulated_data[simulated_data['Player'] == player_name]['Cluster'].iloc[0]

    # Now that cluster_stats is defined, you can safely get the relevant stats for the target player
    target_cluster_stats = cluster_stats.get(target_player_cluster, [])

    # Fetch target player's stats based on their cluster from pitcher_data
    target_player_data = pitcher_data[pitcher_data['Player'] == player_name]

    if not target_player_data.empty:
        # Fetch the relevant stats for the target player from pitcher_data based on their cluster
        target_player_stats = target_player_data[target_cluster_stats]

        # Create a dictionary for target player stats (without Year)
        target_player_dict = {
            'Similar Player': player_name,
        }

        # Add the stats for the target player to the dictionary
        for stat in target_cluster_stats:
            target_player_dict[stat] = target_player_stats[stat].values[0] if not target_player_stats.empty else 'N/A'

        # Display the target player stats table above the similarity table
        target_player_df = pd.DataFrame([target_player_dict])
        st.write(f" ### {player_name} 2024 Season Stats")
        st.markdown(target_player_df.to_html(index=False, escape=False), unsafe_allow_html=True)

    # Initialize list for similar players
    similar_players = []

    # Find the row that matches the Target Player (player_name)
    target_row = similarity_data[similarity_data['Target Player'] == player_name]

    if not target_row.empty:
        # Loop through the columns that contain similar players and their years
        for i in range(1, len(target_row.columns), 2):  # Skip 'Target Player' column, and step through similar player and year columns
            similar_player = target_row.iloc[0, i]  # Get the similar player name
            similar_year = target_row.iloc[0, i + 1]  # Get the year for that similar player

            # Get corresponding data for the similar player and year from Combined
            similar_player_data = combined_data[(combined_data['Player'] == similar_player) & 
                                                (combined_data['Year'] == similar_year)]

            if not similar_player_data.empty:
                # Fetch AAV and Length from the Combined dataset
                aav = similar_player_data['AAV'].values[0]
                length = similar_player_data['Length'].values[0]
                
                # Fetch additional stats for similar player based on cluster
                similar_player_stats = similar_player_data[target_cluster_stats]

                similar_player_dict = {
                    'Similar Player': similar_player,
                    'Year': similar_year,
                    'AAV': aav, 
                    'Length': length
                }

                # Add the stats to the dictionary
                for stat in target_cluster_stats:
                    similar_player_dict[stat] = similar_player_stats[stat].values[0] if not similar_player_stats.empty else 'N/A'

                similar_players.append(similar_player_dict)

    # Display the Similarity Table with stats
    similarity_df = pd.DataFrame(similar_players)

    # Reorder the columns as 'Player', 'Year', {stats}, 'AAV', 'Length'
    ordered_columns = ['Similar Player', 'Year'] + target_cluster_stats + ['AAV', 'Length']
    similarity_df = similarity_df[ordered_columns]

    # Round the values in 'AAV' column to 2 decimal places
    similarity_df['AAV'] = similarity_df['AAV'].apply(lambda x: f"${int(x):,}" if pd.notnull(x) else 'N/A')
    similarity_df['Length'] = similarity_df['Length'].apply(lambda x: f"{int(x)} years" if pd.notnull(x) else 'N/A')

    # Display the reordered similarity table
    st.write("### 5 Most Similar Platform Seasons")
    st.markdown(similarity_df.to_html(index=False, escape=False), unsafe_allow_html=True)



