import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Function for Normalizing the data (Min-Max Scaling)
def normalize_data(df):
    scaler = MinMaxScaler()
    norm_df = scaler.fit_transform(df.iloc[:, 1:])  # Normalize all columns except the first one (alternatives)
    norm_df = pd.DataFrame(norm_df, columns=df.columns[1:])
    return norm_df

# Function for Weighted Normalization
def weighted_normalization(norm_df, weights):
    weighted_matrix = norm_df * weights
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
    pos_distance = np.sqrt(((weighted_matrix - pis) ** 2).sum(axis=1))  # Si+ (Distance to PIS)
    neg_distance = np.sqrt(((weighted_matrix - nis) ** 2).sum(axis=1))  # Si- (Distance to NIS)
    return pos_distance, neg_distance

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
    st.write("Step 1: Normalized Data", pd.concat([df.iloc[:, 0], normalized_df], axis=1))

    # Step 2: Weighted Normalization
    weighted_matrix = weighted_normalization(normalized_df, weights)
    st.write("Step 2: Weighted Normalized Matrix", pd.concat([df.iloc[:, 0], weighted_matrix], axis=1))
    
    # Step 3: Calculate Ideal and Negative Ideal Solutions (A+ and A-)
    pis, nis = calculate_ideal_solutions(weighted_matrix, impacts[0])  # Assuming impacts are the same for all criteria
    st.write("Step 3: Positive Ideal Solution (A+)", pis)
    st.write("Step 3: Negative Ideal Solution (A-)", nis)
    
    # Step 4: Calculate Distances to PIS and NIS (Si+ and Si-)
    pos_distance, neg_distance = calculate_distances(weighted_matrix, pis, nis)
    distance_df = pd.DataFrame({
        'Alternative': df.iloc[:, 0],  # Ensure alternatives column is included
        'Si+ (Distance to PIS)': pos_distance,
        'Si- (Distance to NIS)': neg_distance
    })
    st.write("Step 4: Distances to PIS and NIS", distance_df)
    
    # Step 5: Calculate TOPSIS Scores
    topsis_score = calculate_topsis_score(pos_distance, neg_distance)
    topsis_df = pd.DataFrame({
        'Alternative': df.iloc[:, 0],  # Ensure alternatives column is included
        'TOPSIS Score': topsis_score
    })
    st.write("Step 5: TOPSIS Scores", topsis_df)
    
    # Add the TOPSIS Score to the DataFrame and handle NaN values
    df['TOPSIS Score'] = topsis_score
    df = df.dropna(subset=['TOPSIS Score'])  # Drop rows with NaN TOPSIS Score
    
    # Rank the alternatives based on TOPSIS Score
    df['Rank'] = df['TOPSIS Score'].rank(ascending=False, method='min')  # Rank the alternatives based on TOPSIS Score
    
    # Reset the index and adjust the rank to start from 1
    df['Rank'] = df['Rank'].astype(int)  # Ensure Rank is an integer type
    df = df.reset_index(drop=True)  # Reset the index to start from 0
    
    # Debugging: Check the columns in the DataFrame
    st.write("Columns in the DataFrame:", df.columns)
    
    # Display the final result: Alternatives, TOPSIS Score, Rank
    st.write("Final Results:", df[['Alternative', 'TOPSIS Score', 'Rank']])
    
    # Ensure sorting by Rank before charting
    df_sorted = df[['Alternative', 'TOPSIS Score', 'Rank']].sort_values(by='Rank')
    
    # Displaying the bar chart of TOPSIS scores
    st.bar_chart(df_sorted.set_index('Alternative')['TOPSIS Score'])  # Plot only the TOPSIS Score column
