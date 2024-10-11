from src.neuralnetwork import NeuralNetwork
import streamlit as st

@st.dialog("Data not found")
def data_not_found():
    st.write("Please upload the data first")
    if st.button("Upload Data"):
        st.switch_page("pages/data_handler.py")

if st.button("Train Network", disabled="trainingDF" not in st.session_state):
    nn = NeuralNetwork(30, 40, 4)
    first = True
    for ep in nn.train(st.session_state.trainingDF, st.session_state.validationDF, 0.001, 100):
        if first:
            my_table = st.table(ep)
            first = False
        else:
            my_table.add_rows(ep)
    st.session_state["nn"] = nn
if "nn" in st.session_state:
    st.pyplot(st.session_state.nn.fig)
    with st.container():
        st.write("Now you can predict the results")
        if st.button("Predict", key="predict"):
            st.switch_page("pages/predict.py")
elif "trainingDF" not in st.session_state:
    data_not_found()