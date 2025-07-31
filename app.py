import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Function for TOPSIS calculation
def topsis_method(df, weights, impacts):
    # Normalize the decision matrix
    scaler = MinMaxScaler()
    norm_df = scaler.fit_transform(df.iloc[:, 1:])
    norm_df = pd.DataFrame(norm_df, columns=df.columns[1:])
    
    # Weighted normalized matrix
    weighted_matrix = norm_df * weights
    
    # Positive and negative ideal solutions
    pos_ideal = weighted_matrix.max()
    neg_ideal = weighted_matrix.min()
    
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
        df = pd.read_excel(uploaded_file, engine='openpyxl')  # Added openpyxl engine for Excel files
    
    st.write("Data Preview:", df.head())
    
    # User input for weights and impacts
    st.sidebar.header("Set Criteria Weights")
    criteria = df.columns[1:]
    weights = []
    for criterion in criteria:
        weight = st.sidebar.slider(f"Weight for {criterion}", 0.0, 1.0, 0.1)
        weights.append(weight)
    
    impacts = ['Positive'] * len(criteria)  # Assuming all impacts are positive for simplicity
    weights = np.array(weights)
    
    # Compute TOPSIS scores
    if st.button('Calculate Rankings'):
        df['TOPSIS Score'] = topsis_method(df, weights, impacts)
        df['Rank'] = df['TOPSIS Score'].rank(ascending=False)
        st.write("Results:", df[['Name', 'TOPSIS Score', 'Rank']])
        st.bar_chart(df[['TOPSIS Score']].sort_values('Rank'))
