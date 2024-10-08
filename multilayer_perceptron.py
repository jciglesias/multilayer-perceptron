import streamlit as st
import os

def isData():
    if "validationDF" not in st.session_state:
        return False
    if "trainingDF" not in st.session_state:
        return False
    if "testDF" not in st.session_state:
        return False
    return True

st.title('Multilayer Perceptron')
st.write('''
        This is a simple implementation of a multilayer perceptron with backpropagation and validation. 
        The model is trained on the breast cancer dataset from the University of Wisconsin. 
        The model is trained on 70% of the data, validated on 15% and tested on the remaining 15%.
        ''')

col1, col2 = st.columns(2)
if col1.button("Upload Data", disabled=isData()):
    st.switch_page("pages/data_handler.py")
if col2.button("Erase Data", disabled=not isData()):
    os.system("rm data/*")
if col1.button("Train Model", disabled="nn" in st.session_state):
    st.switch_page("pages/train_network.py")
if col1.button("Predict"):
    st.switch_page("pages/predict.py")