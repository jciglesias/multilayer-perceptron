from src.neuralnetwork import NeuralNetwork
import pandas as pd
import streamlit as st
import pickle, math

interpretation = {
    0: "M",
    1: "B"
}
@st.dialog("Data not found")
def data_not_found():
    st.write("Please upload the data first")
    if st.button("Upload Data"):
        st.switch_page("pages/data_handler.py")

def binary_cross_entropy(y, y_hat, epsilon=0.001):
    y = max(epsilon, min(1 - epsilon, y))
    return -y_hat * math.log(y) - (1 - y_hat) * math.log(1 - y)

if "testDF" in st.session_state:
    pkl_file = open('model.pkl', 'rb')
    nn:NeuralNetwork = pickle.load(pkl_file)
    e = 0
    diff = 0
    for i, row in st.session_state.testingDF.iterrows():
        inputs = row[2:]
        e += binary_cross_entropy(nn.forwardpropagation(inputs), row.iloc[1] == "B")
        if (interpretation[nn.forwardpropagation(inputs)] != row.iloc[1]):
            diff += 1
    e /= len(st.session_state.testingDF)
    print(f"Differences = {diff}/{i}\nBinary cross entropy: {e}")
else:
    data_not_found()