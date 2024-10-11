from src.neuralnetwork import NeuralNetwork
import pandas as pd
import streamlit as st
import math

interpretation = {
    0: "M",
    1: "B"
}
@st.dialog("Data not found")
def data_not_found():
    st.write("Please upload the data first")
    if st.button("Upload Data"):
        st.switch_page("pages/data_handler.py")

@st.dialog("Model not found")
def train_nn():
    st.write("Please train the model first")
    if st.button("Train Model"):
        st.switch_page("pages/train_network.py")

def binary_cross_entropy(y, y_hat, epsilon=0.001):
    y = max(epsilon, min(1 - epsilon, y))
    return -y_hat * math.log(y) - (1 - y_hat) * math.log(1 - y)

if "nn" in st.session_state:
    nn = st.session_state.nn
    e = 0
    diff = 0
    col1, col2 = st.columns(2)
    cont1 = col1.container(height=200)
    cont2 = col2.container(height=200)
    for i, row in st.session_state.testDF.iterrows():
        inputs = row[2:]
        output = nn.forwardpropagation(inputs)
        expected = row.iloc[1] == "B"
        e += binary_cross_entropy(output, expected)
        if i % 2:
            cont1.write(f"Expected: {interpretation[expected]}, got: {interpretation[output]}")
        else:
            cont2.write(f"Expected: {interpretation[expected]}, got: {interpretation[output]}")
        if (interpretation[output] != row.iloc[1]):
            diff += 1
    e /= len(st.session_state.testDF)
    st.write(f"Differences = {diff}/{i}\nBinary cross entropy: {e}")
elif "testDF" in st.session_state:
    train_nn()
else:
    data_not_found()