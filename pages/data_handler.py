import pandas as pd
import streamlit as st


data_file = st.file_uploader("Upload a CSV file", type=["csv"])

if "data" in st.session_state:
    training_tab, validation_tab, test_tab = st.tabs(["Training Data", "Validation Data", "Test Data"])
    with training_tab:
        with st.container(height=400):
            st.table(st.session_state.trainingDF)
    with validation_tab:
        with st.container(height=400):
            st.table(st.session_state.validationDF)
    with test_tab:
        with st.container(height=400):
            st.table(st.session_state.testDF)

if data_file is not None:
    try:
        st.session_state["data"] = pd.read_csv(data_file)
        st.session_state["trainingDF"] = st.session_state.data.sample(frac=0.7)
        st.session_state["validationDF"] = st.session_state.data.drop(st.session_state.trainingDF.index).sample(frac=0.5)
        st.session_state["testDF"] = st.session_state.data.drop(st.session_state.trainingDF.index).drop(st.session_state.validationDF.index)

        st.write("Data uploaded successfully")
        # st.switch_page("multilayer_perceptron.py")
    except Exception as e:
        st.write(f"An error occurred: {e}")

