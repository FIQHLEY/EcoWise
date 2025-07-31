import streamlit as st
import pandas as pd
import numpy as np

# Function to calculate TOPSIS and show each step
def topsis_method(data, weights, impacts):
    # Step 1: Normalize the decision matrix
    norm_data = data / np.sqrt((data ** 2).sum(axis=0))
    st.subheader("Step 1: Normalized Decision Matrix")
    st.write(norm_data)

    # Step 2: Weighted normalized decision matrix
    weighted_data = norm_data * weights
    st.subheader("Step 2: Weighted Normalized Decision Matrix")
    st.write(weighted_data)

    # Step 3: Ideal and negative-ideal solutions
    ideal_solution = np.zeros(data.shape[1])
    negative_ideal_solution = np.zeros(data.shape[1])

    for j in range(data.shape[1]):
        if impacts[j] == 'Benefit':
            ideal_solution[j] = np.max(weighted_data[:, j])
            negative_ideal_solution[j] = np.min(weighted_data[:, j])
        else:  # For 'Cost' criteria, invert the logic
            ideal_solution[j] = np.min(weighted_data[:, j])
            negative_ideal_solution[j] = np.max(weighted_data[:, j])

    st.subheader("Step 3: Ideal and Negative-Ideal Solutions")
    st.write(f"Ideal Solution: {ideal_solution}")
    st.write(f"Negative-Ideal Solution: {negative_ideal_solution}")

    # Step 4: Calculate the Euclidean distance to the ideal and negative-ideal solutions
    distance_to_ideal = np.sqrt(((weighted_data - ideal_solution) ** 2).sum(axis=1))
    distance_to_negative_ideal = np.sqrt(((weighted_data - negative_ideal_solution) ** 2).sum(axis=1))

    st.subheader("Step 4: Euclidean Distance to Ideal and Negative-Ideal Solutions")
    st.write(f"Distance to Ideal Solution: {distance_to_ideal}")
    st.write(f"Distance to Negative-Ideal Solution: {distance_to_negative_ideal}")

    # Step 5: Calculate the TOPSIS score
    topsis_score = distance_to_negative_ideal / (distance_to_ideal + distance_to_negative_ideal)
    st.subheader("Step 5: TOPSIS Scores")
    st.write(topsis_score)

    return topsis_score

# Streamlit app
st.title('EcoWise: MCDM Sustainability Rankings with TOPSIS Methodology')

st.sidebar.header('Input Criteria for Sustainability Ranking')

# Option to upload a file (CSV or Excel)
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file with your decision matrix", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Check file extension to determine if it's CSV or Excel
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)

    # Display the uploaded decision matrix
    st.subheader('Uploaded Decision Matrix')
    st.dataframe(df)
    
    # Input: Criteria weights
    st.sidebar.subheader('Enter the weights for each criterion')
    weights = []
    for j in range(df.shape[1] - 1):  # Excluding the first column (Alternatives)
        weight = st.sidebar.slider(f'Weight for Criterion {j+1}', min_value=0.0, max_value=1.0, value=1.0, step=0.1)
        weights.append(weight)
    weights = np.array(weights)
    
    # Display a warning if total weight exceeds 1
    if np.sum(weights) > 1.0:
        st.sidebar.warning("The total weight exceeds 1. Please adjust the weights.")

    weights = weights / np.sum(weights)  # Normalize the weights to sum to 1

    # Input: Impact (benefit or cost) for each criterion
    st.sidebar.subheader('Enter the impact type (Benefit or Cost) for each criterion')
    impacts = []
    for j in range(df.shape[1] - 1):  # Excluding the first column (Alternatives)
        impact = st.sidebar.radio(f'Impact for Criterion {j+1}', options=['Benefit', 'Cost'], key=f"impact_{j}")
        impacts.append(impact)
    impacts = np.array(impacts)

    # Calculate TOPSIS score
    if st.sidebar.button('Calculate Rankings'):
        topsis_scores = topsis_method(df.iloc[:, 1:].to_numpy(), weights, impacts)
        
        # Display results
        st.subheader('TOPSIS Ranking Results')
        ranking_df = pd.DataFrame({'Alternative': df.iloc[:, 0], 'TOPSIS Score': topsis_scores})
        ranking_df['Rank'] = ranking_df['TOPSIS Score'].rank(ascending=False)
        ranking_df = ranking_df.sort_values(by='Rank')
        
        st.write(ranking_df)

        # Visualization of the rankings
        st.subheader('Bar Chart of TOPSIS Rankings')
        st.bar_chart(ranking_df.set_index('Alternative')['TOPSIS Score'])
else:
    st.info('Please upload a CSV or Excel file to get started.')
