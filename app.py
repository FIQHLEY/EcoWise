import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Function for TOPSIS calculation
def topsis_method(df, weights, impacts):
    # Normalize the decision matrix (skip the index/first column)
    scaler = MinMaxScaler()
    norm_df = scaler.fit_transform(df.iloc[:, 1:])  # Exclude the first column (alternatives)
    norm_df = pd.DataFrame(norm_df, columns=df.columns[1:])
    
    # Ensure weights align with the criteria columns (C1, C2, C3, ...)
    weighted_matrix = norm_df * weights  # Perform element-wise multiplication
    
    # Positive and negative ideal solutions
    pos_ideal = weighted_matrix.max() if impacts == 'Benefit' else weighted_matrix.min()
    neg_ideal = weighted_matrix.min() if impacts == 'Benefit' else weighted_matrix.max()
    
    # Euclidean distance from ideal solutions
    pos_distance = np.sqrt(((weighted_matrix - pos_ideal) ** 2).sum(axis=1))
    neg_distance = np.sqrt(((weighted_matrix - neg_ideal) ** 2).sum(axis=1))
    
    # Calculate the TOPSIS score
    topsis_score = neg_distance / (pos_distance + neg_distance)
    
    return topsis_score

# Streamlit UI
st.title('EcoWise: MCDM Sustainability Rankings with TOPSIS Methodology')

# File upload
uploaded_file = st.file_uploader("Upload your data", type=["csv", "xlsx"])
if uploaded_file is not None:
    if uploaded_file.name.endswith('csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    
    # Correcting the reading of data
    if 'Unnamed: 0' in df.columns:
        df.set_index('Unnamed: 0', inplace=True)
    
    st.write("Data Preview:", df.head())
    
    # Setting weight sliders for each criterion (no normalization or sum to 1)
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
    
    # Compute TOPSIS scores
    if st.button('Calculate Rankings'):
        df['TOPSIS Score'] = topsis_method(df, weights, impacts)
        df['Rank'] = df['TOPSIS Score'].rank(ascending=False)  # Rank the alternatives based on TOPSIS Score
        st.write("Results:", df[['TOPSIS Score', 'Rank']])
        
        # Ensure sorting by Rank before charting
        df_sorted = df[['TOPSIS Score', 'Rank']].sort_values(by='Rank')
        
        st.bar_chart(df_sorted['TOPSIS Score'])  # Plot only the TOPSIS Score column
