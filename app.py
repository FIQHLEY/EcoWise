import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Function for Normalizing the data (Min-Max Scaling)
def normalize_data(df):
    scaler = MinMaxScaler()
    norm_df = scaler.fit_transform(df.iloc[:, 1:])  # Normalize all columns except the first one (alternatives)
    norm_df = pd.DataFrame(norm_df, columns=df.columns[1:])
    
    # Add alternatives as A1, A2, ..., A14
    norm_df.insert(0, 'Alternatives', [f'A{i+1}' for i in range(len(df))])
    return norm_df

# Function for Weighted Normalization
def weighted_normalization(norm_df, weights):
    weighted_matrix = norm_df.iloc[:, 1:].multiply(weights, axis=1)  # Multiply normalized data by weights
    weighted_matrix.insert(0, 'Alternatives', norm_df['Alternatives'])  # Add Alternatives column back
    return weighted_matrix

# Function to calculate Ideal Solutions (A+ and A-)
def calculate_ideal_solutions(weighted_matrix, impacts):
    if impacts == 'Benefit':
        pis = weighted_matrix.max()  # Positive Ideal Solution (Best values for benefits)
        nis = weighted_matrix.min()  # Negative Ideal Solution (Worst values for benefits)
    else:
        pis = weighted_matrix.min()  # Positive Ideal Solution (Best values for costs)
        nis = weighted_matrix.max()  # Negative Ideal Solution (Worst values for costs)
    
    return pis, nis

# Function to calculate the Euclidean Distances (Si+ and Si-)
def calculate_distances(weighted_matrix, pis, nis):
    # Ensure pis and nis are numpy arrays and broadcast them for element-wise subtraction
    pis = np.array(pis)
    nis = np.array(nis)
    
    # Calculate the Euclidean distance to the PIS and NIS
    pos_distance = np.sqrt(((weighted_matrix.iloc[:, 1:].values - pis) ** 2).sum(axis=1))  # Si+ (Distance to PIS)
    neg_distance = np.sqrt(((weighted_matrix.iloc[:, 1:].values - nis) ** 2).sum(axis=1))  # Si- (Distance to NIS)
    
    weighted_matrix['Si+'] = pos_distance  # Add Si+ to the weighted matrix
    weighted_matrix['Si-'] = neg_distance  # Add Si- to the weighted matrix
    return pos_distance, neg_distance, weighted_matrix

# Function to calculate the TOPSIS Scores
def calculate_topsis_score(pos_distance, neg_distance):
    topsis_score = neg_distance / (pos_distance + neg_distance)  # TOPSIS Score formula
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
    
    # Step 3: Calculate Ideal and Negative Ideal Solutions (A+ and A-)
    pis, nis = calculate_ideal_solutions(weighted_matrix, impacts[0])  # Assuming impacts are the same for all criteria
    st.write("Step 3: Positive Ideal Solution (A+)", pis)
    st.write("Step 3: Negative Ideal Solution (A-)", nis)
    
    # Step 4: Calculate Distances to PIS and NIS (Si+ and Si-)
    pos_distance, neg_distance, weighted_matrix_with_distances = calculate_distances(weighted_matrix, pis, nis)
    st.write("Step 4: Distances to PIS and NIS", weighted_matrix_with_distances)
    
    # Step 5: Calculate TOPSIS Scores
    topsis_score = calculate_topsis_score(pos_distance, neg_distance)
    st.write("Step 5: TOPSIS Scores", topsis_score)
    
    # Add the TOPSIS Score to the DataFrame and handle NaN values
    weighted_matrix_with_distances['TOPSIS Score'] = topsis_score
    weighted_matrix_with_distances = weighted_matrix_with_distances.dropna(subset=['TOPSIS Score'])  # Drop rows with NaN TOPSIS Score
    
    # Rank the alternatives based on TOPSIS Score
    weighted_matrix_with_distances['Rank'] = weighted_matrix_with_distances['TOPSIS Score'].rank(ascending=False, method='min')  # Rank the alternatives based on TOPSIS Score
    
    # Ensure index starts from 1
    weighted_matrix_with_distances['Rank'] = weighted_matrix_with_distances['Rank'].astype(int)
    
    st.write("Final Results:", weighted_matrix_with_distances[['Alternatives', 'TOPSIS Score', 'Rank']])
    
    # Ensure sorting by Rank before charting
    df_sorted = weighted_matrix_with_distances[['Alternatives', 'TOPSIS Score', 'Rank']].sort_values(by='Rank')
    
    # Displaying the bar chart of TOPSIS scores
    st.bar_chart(df_sorted['TOPSIS Score'])  # Plot only the TOPSIS Score column
