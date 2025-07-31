import streamlit as st
import pandas as pd
import numpy as np

# Function to calculate TOPSIS
def topsis_method(data, weights, impacts):
    # Normalize the decision matrix
    norm_data = data / np.sqrt((data ** 2).sum(axis=0))

    # Weighted normalized decision matrix
    weighted_data = norm_data * weights

    # Ideal and negative-ideal solutions
    ideal_solution = np.max(weighted_data, axis=0) if impacts == 'Benefit' else np.min(weighted_data, axis=0)
    negative_ideal_solution = np.min(weighted_data, axis=0) if impacts == 'Benefit' else np.max(weighted_data, axis=0)

    # Calculate the Euclidean distance to the ideal and negative-ideal solutions
    distance_to_ideal = np.sqrt(((weighted_data - ideal_solution) ** 2).sum(axis=1))
    distance_to_negative_ideal = np.sqrt(((weighted_data - negative_ideal_solution) ** 2).sum(axis=1))

    # Calculate the TOPSIS score
    topsis_score = distance_to_negative_ideal / (distance_to_ideal + distance_to_negative_ideal)
    
    return topsis_score

# Streamlit app
st.title('EcoWise: MCDM Sustainability Rankings with TOPSIS Methodology')

st.sidebar.header('Input Criteria for Sustainability Ranking')

# Input: Number of alternatives
num_alternatives = st.sidebar.number_input('Enter the number of alternatives (projects/entities)', min_value=1, step=1)

# Input: Number of criteria
num_criteria = st.sidebar.number_input('Enter the number of criteria for evaluation', min_value=1, step=1)

# Input: Data for decision matrix
st.sidebar.subheader('Enter the data for the decision matrix (alternatives x criteria)')
decision_matrix = []
for i in range(num_alternatives):
    row = []
    for j in range(num_criteria):
        row.append(st.sidebar.number_input(f'Value for Alternative {i+1}, Criteria {j+1}', min_value=0.0, step=0.1))
    decision_matrix.append(row)

# Convert decision matrix to DataFrame
df = pd.DataFrame(decision_matrix, columns=[f'Criterion {i+1}' for i in range(num_criteria)], index=[f'Alternative {i+1}' for i in range(num_alternatives)])

# Input: Criteria weights
st.sidebar.subheader('Enter the weights for each criterion')
weights = []
for j in range(num_criteria):
    weights.append(st.sidebar.slider(f'Weight for Criterion {j+1}', min_value=0.0, max_value=1.0, value=1.0, step=0.1))
weights = np.array(weights) / np.sum(weights)  # Normalize the weights

# Input: Impact (benefit or cost) for each criterion
st.sidebar.subheader('Enter the impact type (Benefit or Cost) for each criterion')
impacts = []
for j in range(num_criteria):
    impacts.append(st.sidebar.radio(f'Impact for Criterion {j+1}', options=['Benefit', 'Cost']))
impacts = np.array(impacts)

# Calculate TOPSIS score
if st.sidebar.button('Calculate Rankings'):
    topsis_scores = topsis_method(df.to_numpy(), weights, impacts)
    
    # Display results
    st.subheader('TOPSIS Ranking Results')
    ranking_df = pd.DataFrame({'Alternative': df.index, 'TOPSIS Score': topsis_scores})
    ranking_df['Rank'] = ranking_df['TOPSIS Score'].rank(ascending=False)
    ranking_df = ranking_df.sort_values(by='Rank')
    
    st.write(ranking_df)

    # Visualization of the rankings
    st.subheader('Bar Chart of TOPSIS Rankings')
    st.bar_chart(ranking_df.set_index('Alternative')['TOPSIS Score'])

# Display decision matrix
st.subheader('Decision Matrix')
st.dataframe(df)

