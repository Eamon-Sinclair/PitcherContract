import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

#Load Datasets
simulated_data = pd.read_csv("https://raw.githubusercontent.com/Eamon-Sinclair/PitcherContract/main/Sim_Data.csv")
contract_data = pd.read_csv("https://raw.githubusercontent.com/Eamon-Sinclair/PitcherContract/main/FAContract.csv")
similarity_data = pd.read_csv("https://raw.githubusercontent.com/Eamon-Sinclair/PitcherContract/main/Similar_Players.csv")
pitcher_data = pd.read_csv("https://raw.githubusercontent.com/Eamon-Sinclair/PitcherContract/main/Pitcher_Data.csv")
combined_data = pd.read_csv("https://raw.githubusercontent.com/Eamon-Sinclair/PitcherContract/main/Combined.csv")

#Begin Page
st.title("Starting Pitcher Contract Model")

st.markdown("""
    **Created by Eamon Sinclair and Matthew Krakower**  
    **Twitter**: [EamonSinclair](https://x.com/_EamonSinclair), [MatthewKrakower](https://x.com/MatthewKrakower)  
    **Email**: [eamonsinclair15@gmail.com](mailto:eamonsinclair15@gmail.com), [Matthew.krakower21@gmail.com](mailto:Matthew.krakower21@gmail.com)    
    **Linkedin**: [EamonSinclair](https://www.linkedin.com/in/eamonsinclair/), [MatthewKrakower](https://www.linkedin.com/in/matthew-krakower-8b657827b/)         
    **Data**: [Baseball Prospectus](https://www.baseballprospectus.com/)
""", unsafe_allow_html=True)

#Select Player
players = simulated_data['Player'].unique()
player_name = st.selectbox("Select a Player", players)

#Initial Table
if player_name:
    player_data = simulated_data[simulated_data['Player'] == player_name]
    
    lower_aav = player_data['Lower Bound AAV'].iloc[0]
    median_aav = player_data['Median AAV'].iloc[0]
    upper_aav = player_data['Upper Bound AAV'].iloc[0]
    lower_length = player_data['Lower Bound Length'].iloc[0]
    median_length = player_data['Median Length'].iloc[0]
    upper_length = player_data['Upper Bound Length'].iloc[0]

    lower_total = lower_aav * round(lower_length)
    median_total = median_aav * round(median_length)
    upper_total = upper_aav * round(upper_length)

    recent_contract = contract_data[contract_data['Player'] == player_name].sort_values(by='Year', ascending=False).head(1)
    if not recent_contract.empty:
        recent_aav = recent_contract.iloc[0]['AAV']
        recent_length = recent_contract.iloc[0]['Length']
        recent_total = recent_aav * recent_length
        last_contract = [f"${recent_aav:,.2f}", f"{recent_length:.2f} years", f"${recent_total:,.2f}"]
    else:
        last_contract = ["No FA Contract From '21 - '24", "N/A", "N/A"]

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

    predicted_values_html = table_df.to_html(index=False, escape=False)
    predicted_values_html = f'<div class="predicted-table blue-shade">{predicted_values_html}</div>'

    st.markdown(predicted_values_html, unsafe_allow_html=True)

    sim_aav = player_data[[col for col in simulated_data.columns if col.startswith("SimAAV")]].values.flatten()
    sim_length = player_data[[col for col in simulated_data.columns if col.startswith("SimLength")]].values.flatten()

    #Histograms
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))  # 1 row, 2 columns

    sns.kdeplot(sim_aav, fill=True, color="lightblue", label="AAV Distribution", ax=axes[0])
    axes[0].axvline(player_data['Median AAV'].iloc[0], color='red', linestyle='--', label='Median AAV')
    axes[0].axvline(player_data['Lower Bound AAV'].iloc[0], color='black', linestyle='--', label='Lower Bound AAV')  # Lower bound
    axes[0].axvline(player_data['Upper Bound AAV'].iloc[0], color='black', linestyle='--', label='Upper Bound AAV')  # Upper bound
    axes[0].set_xlabel("AAV ($M)", fontsize=20)  # Increase font size for x-axis label
    axes[0].set_ylabel("Density", fontsize=20)  # Increase font size for y-axis label
    axes[0].set_title(f"AAV Distribution for {player_name}", fontsize=24)  # Increase font size for title
    axes[0].legend(loc='upper right', fontsize=12)  # Increase font size for legend

    axes[0].yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    axes[0].yaxis.get_offset_text().set_visible(False)  # Hide offset text if present
    axes[0].ticklabel_format(style='plain', axis='y')

    y_ticks = axes[0].get_yticks()
    axes[0].set_yticklabels([f'{round(tick * 1e7, 1):,}' for tick in y_ticks], fontsize=18)  # Increase y-tick font size

    ticks = axes[0].get_xticks()
    axes[0].set_xticklabels([f'{round(tick / 1_000_000, 2)}' for tick in ticks], fontsize=18)  # Increase x-tick font size

    sns.kdeplot(sim_length, fill=True, color="lightgreen", label="Length Distribution", ax=axes[1])
    axes[1].axvline(player_data['Median Length'].iloc[0], color='red', linestyle='--', label='Median Length')
    axes[1].axvline(player_data['Lower Bound Length'].iloc[0], color='black', linestyle='--', label='Lower Bound Length')  # Lower bound
    axes[1].axvline(player_data['Upper Bound Length'].iloc[0], color='black', linestyle='--', label='Upper Bound Length')  # Upper bound
    axes[1].set_xlabel("Contract Length (Years)", fontsize=20)  # Increase font size for x-axis label
    axes[1].set_ylabel("Density", fontsize=20)  # Increase font size for y-axis label
    axes[1].set_title(f"Length Distribution for {player_name}", fontsize=24)  # Increase font size for title
    axes[1].legend(loc='upper right', fontsize=12)  # Increase font size for legend
   
    y_ticks = axes[1].get_yticks()
    axes[1].set_yticklabels([f'{tick:.2f}' for tick in y_ticks], fontsize=18)  # Increase y-tick font size

    ticks = axes[1].get_xticks()
    axes[1].set_xticklabels([f'{tick:.2f}' for tick in ticks], fontsize=18)  # Increase x-tick font size

    st.pyplot(fig)
    
    #Similarity Table
    cluster_stats = {
        0: ['WARP', 'IPGS', 'GS', 'DRA', 'CSProb', 'Age'],
        1: ['IPGS', 'WHIP', 'Swing', 'RA9', 'DRA', 'Age'],
        2: ['DRA', 'KBB', 'WARP', 'Age', 'ZContact', 'CSProb']
    }

    target_player_cluster = simulated_data[simulated_data['Player'] == player_name]['Cluster'].iloc[0]

    target_cluster_stats = cluster_stats.get(target_player_cluster, [])

    target_player_data = pitcher_data[pitcher_data['Player'] == player_name]

    if not target_player_data.empty:
        # Fetch the relevant stats for the target player from pitcher_data based on their cluster
        target_player_stats = target_player_data[target_cluster_stats]

        target_player_dict = {
            'Similar Player': player_name,
        }

        for stat in target_cluster_stats:
            target_player_dict[stat] = target_player_stats[stat].values[0] if not target_player_stats.empty else 'N/A'

        target_player_df = pd.DataFrame([target_player_dict])
        st.write(f" ### {player_name} 2024 Season Stats")
        st.markdown(target_player_df.to_html(index=False, escape=False), unsafe_allow_html=True)

    similar_players = []

    target_row = similarity_data[similarity_data['Target Player'] == player_name]

    if not target_row.empty:
        # Loop through the columns that contain similar players and their years
        for i in range(1, len(target_row.columns), 2):
            similar_player = target_row.iloc[0, i]  
            similar_year = target_row.iloc[0, i + 1]  

            similar_player_data = combined_data[(combined_data['Player'] == similar_player) & 
                                                (combined_data['Year'] == similar_year)]

            if not similar_player_data.empty:
                aav = similar_player_data['AAV'].values[0]
                length = similar_player_data['Length'].values[0]
                
                similar_player_stats = similar_player_data[target_cluster_stats]

                similar_player_dict = {
                    'Similar Player': similar_player,
                    'Year': similar_year,
                    'AAV': aav, 
                    'Length': length
                }

                for stat in target_cluster_stats:
                    similar_player_dict[stat] = similar_player_stats[stat].values[0] if not similar_player_stats.empty else 'N/A'

                similar_players.append(similar_player_dict)

    similarity_df = pd.DataFrame(similar_players)

    ordered_columns = ['Similar Player', 'Year'] + target_cluster_stats + ['AAV', 'Length']
    similarity_df = similarity_df[ordered_columns]

    similarity_df['AAV'] = similarity_df['AAV'].apply(lambda x: f"${int(x):,}" if pd.notnull(x) else 'N/A')
    similarity_df['Length'] = similarity_df['Length'].apply(lambda x: f"{int(x)} years" if pd.notnull(x) else 'N/A')

    st.write("### 5 Most Similar Platform Seasons")
    st.markdown(similarity_df.to_html(index=False, escape=False), unsafe_allow_html=True)



