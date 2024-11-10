import pickle
import pandas as pd
import streamlit as st

# Load the model and encoders
with open('model_penguin_65130701938.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# Streamlit app
st.title('Penguin Species Prediction')

# Input fields for user data
island = st.selectbox('Select Island', ['Torgersen', 'Biscoe', 'Dream'])
culmen_length_mm = st.number_input('Culmen Length (mm)', min_value=0.0)
culmen_depth_mm = st.number_input('Culmen Depth (mm)', min_value=0.0)
flipper_length_mm = st.number_input('Flipper Length (mm)', min_value=0.0)
body_mass_g = st.number_input('Body Mass (g)', min_value=0)
sex = st.selectbox('Select Sex', ['MALE', 'FEMALE'])

# When the user clicks the 'Predict' button
if st.button('Predict Species'):
    # Prepare the input data
    x_new = pd.DataFrame({
        'island': [island],
        'culmen_length_mm': [culmen_length_mm],
        'culmen_depth_mm': [culmen_depth_mm],
        'flipper_length_mm': [flipper_length_mm],
        'body_mass_g': [body_mass_g],
        'sex': [sex]
    })
    
    # Apply encoding for categorical features
    x_new['island'] = island_encoder.transform(x_new['island'])
    x_new['sex'] = sex_encoder.transform(x_new['sex'])
    
    # Make prediction
    y_pred_new = model.predict(x_new)
    
    # Decode the predicted species
    result = species_encoder.inverse_transform(y_pred_new)
    
    # Display the prediction result
    st.write(f'Predicted Species: {result[0]}')

