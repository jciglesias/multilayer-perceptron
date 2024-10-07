import pandas as pd
import streamlit as st

st.session_state["data_file"] = st.file_uploader("Upload a CSV file", type=["csv"])

if st.session_state.data_file is not None:
    try:
        data = pd.read_csv(st.session_state.data_file)
        st.session_state["trainingDF"] = data.sample(frac=0.7)
        st.session_state["validationDF"] = data.drop(st.session_state.trainingDF.index).sample(frac=0.5)
        st.session_state["testDF"] = data.drop(st.session_state.trainingDF.index).drop(st.session_state.validationDF.index)

        # training.to_csv("data/training.csv", index=False)
        # validation.to_csv("data/validation.csv", index=False)
        # test.to_csv("data/test.csv", index=False)
        st.write("Data uploaded successfully")
        st.switch_page("multilayer_perceptron.py")
    except Exception as e:
        st.write(f"An error occurred: {e}")