from src.neuralnetwork import NeuralNetwork
import streamlit as st

@st.dialog("Data not found")
def data_not_found():
    st.write("Please upload the data first")
    if st.button("Upload Data"):
        st.switch_page("pages/data_handler.py")

if st.button("Train Network", disabled="trainingDF" not in st.session_state):
    nn = NeuralNetwork(30, 40, 4)
    with st.container(border=True, height=200):
        for ep in nn.train(st.session_state.trainingDF, st.session_state.validationDF, 0.001, 100):
            st.write(ep)
    st.session_state["nn"] = nn
elif "trainingDF" not in st.session_state:
    data_not_found()
if "nn" in st.session_state:
    st.pyplot(st.session_state.nn.fig)