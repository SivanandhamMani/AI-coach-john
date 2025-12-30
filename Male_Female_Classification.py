import pandas as pd
import streamlit as st 
from pickle import load

st.title('Model Deployment: Logistic Regression')

st.sidebar.header('User Input Parameters')

def user_input_features():
    height = st.sidebar.number_input("Height")
    weight = st.sidebar.number_input("Weight")
    data = {'Height_cm':height,
            'Weight':weight}
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

# load the model from disk
loaded_model = load(open('gender_classification_intelligence.pkl', 'rb'))

prediction = loaded_model.predict(df)
prediction_proba = loaded_model.predict_proba(df)

st.subheader('Predicted Result')
st.write('Female' if prediction_proba[0][1] > 0.5 else 'Male')

st.subheader('Prediction Probability')
st.write(prediction_proba)


