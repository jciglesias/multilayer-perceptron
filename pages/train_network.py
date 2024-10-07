from src.neuralnetwork import NeuralNetwork
import streamlit as st

@st.dialog("Data not found")
def data_not_found():
    st.write("Please upload the data first")
    if st.button("Upload Data"):
        st.switch_page("pages/data_handler.py")

try:
    # data = pd.read_csv('data/training.csv')
    # val_data = pd.read_csv('data/validation.csv')
    nn = NeuralNetwork(30, 40, 4)
    nn.train(st.session_state.trainingDF, st.session_state.validationDF, 0.001, 100)
    print(nn)
    nn.save('model.pkl')
except Exception as e:
    data_not_found()