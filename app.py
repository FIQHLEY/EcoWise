import streamlit as st
import pandas as pd
import numpy as np

# Function to calculate TOPSIS with step-by-step results
def topsis_method(data, weights, impacts):
    # Step 1: Normalize the decision matrix
    norm_data = data / np.sqrt((data ** 2).sum(axis=0))
    
    # Step 2: Weighted normalized decision matrix
    weighted_data = norm_data * weights
    
    # Step 3: Calculate the ideal and negative-ideal solutions
    ideal_solution = np.zeros(data.shape[1])
    negative_ideal_solution = np.zeros(data.shape[1])
    
    for j in range(data.shape[1]):
        if impacts[j] == 'Benefit':
            ideal_solution[j] = np.max(weighted_data[:, j])
            negative_ideal_solution[j] = np.min(weighted_data[:, j])
        else:  # For 'Cost' criteria, invert the logic
            ideal_solution[j] = np.min(weighted_data[:, j])
            negative_ideal_solution[j] = np.max(weighted_data[:, j])

    # Step 4: Calculate the Euclidean distance to the ideal and negative-ideal solutions
    distance_to_ideal = np.sqrt(((weighted_data - ideal_solution) ** 2).sum(axis=1))
    distance_to_negative_ideal = np.sqrt(((weighted_data - negative_ideal_solution) ** 2).sum(axis=1))

    # Step 5: Calculate the TOPSIS score
    topsis_score = distance_to_negative_ideal / (distance_to_ideal + distance_to_negative_ideal)
    
    return norm_data, weighted_data, ideal_solution, negative_ideal_solution, distance_to_ideal, distance_to_negative_ideal, topsis_score

# Streamlit app
st.title('EcoWise: Web-based Intelligent Sustainability Evaluation using TOPSIS (WISE-T)')

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
    
    # Step 1: Input: Criteria weights
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

    # Step 2: Input: Impact (benefit or cost) for each criterion
    st.sidebar.subheader('Enter the impact type (Benefit or Cost) for each criterion')
    impacts = []
    for j in range(df.shape[1] - 1):  # Excluding the first column (Alternatives)
        impact = st.sidebar.radio(f'Impact for Criterion {j+1}', options=['Benefit', 'Cost'], key=f"impact_{j}")
        impacts.append(impact)
    impacts = np.array(impacts)

    # Step 3: Calculate TOPSIS step-by-step results
    if st.sidebar.button('Calculate Rankings'):
        norm_data, weighted_data, ideal_solution, negative_ideal_solution, distance_to_ideal, distance_to_negative_ideal, topsis_scores = topsis_method(df.iloc[:, 1:].to_numpy(), weights, impacts)
        
        # Step 4: Display normalization results
        st.subheader('Step 1: Normalized Decision Matrix')
        norm_df = pd.DataFrame(norm_data, columns=[f'Criterion {i+1}' for i in range(df.shape[1] - 1)], index=df.iloc[:, 0])
        st.write(norm_df)

        # Step 5: Display weighted normalization results
        st.subheader('Step 2: Weighted Normalized Decision Matrix')
        weighted_df = pd.DataFrame(weighted_data, columns=[f'Criterion {i+1}' for i in range(df.shape[1] - 1)], index=df.iloc[:, 0])
        st.write(weighted_df)

        # Step 6: Display Ideal and Negative-Ideal Solutions in a Table
        st.subheader('Step 3: Ideal and Negative-Ideal Solutions')

        # Create a DataFrame to display Ideal and Negative-Ideal solutions in a table
        ideal_negative_ideal_df = pd.DataFrame({
            'Criterion': [f'Criterion {i+1}' for i in range(df.shape[1] - 1)],
            'Impact': impacts,
            'Ideal Solution (A+)': ideal_solution,
            'Negative-Ideal Solution (A-)': negative_ideal_solution
        })

        st.write(ideal_negative_ideal_df)

        # Step 7: Display the Euclidean distances
        st.subheader('Step 4: Euclidean Distance to Ideal and Negative-Ideal Solutions')
        distance_df = pd.DataFrame({
            'Alternative': df.iloc[:, 0],
            'Distance to Ideal (D+)': distance_to_ideal,
            'Distance to Negative-Ideal (D-)': distance_to_negative_ideal
        })
        st.write(distance_df)

        # Step 8: Display TOPSIS Ranking Results
        st.subheader('Step 5: TOPSIS Ranking Results')
        ranking_df = pd.DataFrame({'Alternative': df.iloc[:, 0], 'TOPSIS Score': topsis_scores})
        ranking_df['Rank'] = ranking_df['TOPSIS Score'].rank(ascending=False)
        ranking_df = ranking_df.sort_values(by='Rank')
        st.write(ranking_df)

        # Step 9: Visualization of the rankings
        st.subheader('Bar Chart of TOPSIS Rankings')
        st.bar_chart(ranking_df.set_index('Alternative')['TOPSIS Score'])

else:
    st.info('Please upload a CSV or Excel file to get started.')
