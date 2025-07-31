import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Step 1: Data Normalization
def normalize_data(df):
    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df.iloc[:, 1:])  # Normalize all columns except the index (alternatives)
    normalized_df = pd.DataFrame(normalized_data, columns=df.columns[1:])
    
    return normalized_df

# Step 2: Weighted Normalization
def weighted_normalization(normalized_df, weights):
    # Element-wise multiplication of the normalized data by weights
    weighted_matrix = normalized_df * weights
    return weighted_matrix

# Step 3: Calculate Ideal and Negative Ideal Solutions
def calculate_ideal_solutions(weighted_matrix, impacts):
    # Positive Ideal Solution (PIS) and Negative Ideal Solution (NIS)
    if impacts == 'Benefit':
        pis = weighted_matrix.max()  # Best values for benefit criteria
        nis = weighted_matrix.min()  # Worst values for benefit criteria
    else:
        pis = weighted_matrix.min()  # Best values for cost criteria
        nis = weighted_matrix.max()  # Worst values for cost criteria
        
    return pis, nis

# Step 4: Calculate the Euclidean Distance to PIS and NIS
def calculate_distances(weighted_matrix, pis, nis):
    # Euclidean distance from PIS and NIS
    pos_distance = np.sqrt(((weighted_matrix - pis) ** 2).sum(axis=1))
    neg_distance = np.sqrt(((weighted_matrix - nis) ** 2).sum(axis=1))
    
    return pos_distance, neg_distance

# Step 5: Calculate TOPSIS Scores
def calculate_topsis_score(pos_distance, neg_distance):
    # TOPSIS score calculation
    topsis_score = neg_distance / (pos_distance + neg_distance)
    return topsis_score

# Streamlit UI for input and result display
st.title('Step-by-Step TOPSIS Method for Sustainability Rankings')

# File upload
uploaded_file = st.file_uploader("Upload your data", type=["csv", "xlsx"])
if uploaded_file is not None:
    if uploaded_file.name.endswith('csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    
    # Display the dataset preview
    st.write("Data Preview:", df.head())
    
    # Setting weight sliders for each criterion
    st.sidebar.header("Set Criteria Weights")
    criteria = df.columns[1:]  # All columns except the first column (alternatives)
    weights = []  # Initialize an empty list for weights
    
    # Create a slider for each criterion (no weight sum restriction)
    for criterion in criteria:
        weight = st.sidebar.slider(f"Weight for {criterion}", 0.0, 1.0, 0.0)
        weights.append(weight)
    
    weights = np.array(weights)  # Convert weights to a numpy array
    
    # User input for impacts (Benefit or Cost)
    impacts = []
    for criterion in criteria:
        impact = st.sidebar.selectbox(f"Is {criterion} a benefit or cost?", ['Benefit', 'Cost'], key=criterion)
        impacts.append(impact)
    
    # Step 1: Normalize the data
    normalized_df = normalize_data(df)
    st.write("Step 1: Normalized Data", normalized_df)

    # Step 2: Weighted Normalization
    weighted_matrix = weighted_normalization(normalized_df, weights)
    st.write("Step 2: Weighted Normalized Matrix", weighted_matrix)
    
    # Step 3: Calculate Ideal and Negative Ideal Solutions
    pis, nis = calculate_ideal_solutions(weighted_matrix, impacts[0])  # Assuming impacts are the same for all criteria
    st.write("Step 3: Positive Ideal Solution (PIS)", pis)
    st.write("Step 3: Negative Ideal Solution (NIS)", nis)
    
    # Step 4: Calculate Distances to PIS and NIS
    pos_distance, neg_distance = calculate_distances(weighted_matrix, pis, nis)
    st.write("Step 4: Positive Ideal Solution Distance", pos_distance)
    st.write("Step 4: Negative Ideal Solution Distance", neg_distance)
    
    # Step 5: Calculate TOPSIS Scores
    topsis_score = calculate_topsis_score(pos_distance, neg_distance)
    st.write("Step 5: TOPSIS Scores", topsis_score)
    
    # Rank the alternatives based on TOPSIS Score
    df['TOPSIS Score'] = topsis_score
    df['Rank'] = df['TOPSIS Score'].rank(ascending=False)  # Rank the alternatives based on TOPSIS Score
    st.write("Final Results:", df[['TOPSIS Score', 'Rank']])
    
    # Displaying the bar chart of TOPSIS scores
    st.bar_chart(df[['TOPSIS Score']].sort_values('Rank'))
