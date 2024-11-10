

import streamlit as st
import pickle
import pandas as pd

# Function to load the model and encoders from uploaded files
def load_model_and_encoders(model_file, species_encoder_file, island_encoder_file, sex_encoder_file):
    model = pickle.load(model_file)
    species_encoder = pickle.load(species_encoder_file)
    island_encoder = pickle.load(island_encoder_file)
    sex_encoder = pickle.load(sex_encoder_file)
    return model, species_encoder, island_encoder, sex_encoder

# Streamlit UI setup
st.title('Penguin Species Prediction')

# File uploads
model_file = st.file_uploader('Upload the model (model_penguin_65130701938.pkl)', type='pkl')
species_encoder_file = st.file_uploader('Upload the species encoder (species_encoder.pkl)', type='pkl')
island_encoder_file = st.file_uploader('Upload the island encoder (island_encoder.pkl)', type='pkl')
sex_encoder_file = st.file_uploader('Upload the sex encoder (sex_encoder.pkl)', type='pkl')

# User inputs for prediction
island = st.selectbox('Island', ['Torgersen', 'Biscoe', 'Dream'])
culmen_length = st.number_input('Culmen Length (mm)', min_value=30.0, max_value=70.0)
culmen_depth = st.number_input('Culmen Depth (mm)', min_value=10.0, max_value=30.0)
flipper_length = st.number_input('Flipper Length (mm)', min_value=150.0, max_value=250.0)
body_mass = st.number_input('Body Mass (g)', min_value=2500.0, max_value=7000.0)
sex = st.selectbox('Sex', ['Male', 'Female'])

# When the "Predict" button is pressed
if st.button('Predict'):
    if model_file and species_encoder_file and island_encoder_file and sex_encoder_file:
        # Load the model and encoders
        model, species_encoder, island_encoder, sex_encoder = load_model_and_encoders(
            model_file, species_encoder_file, island_encoder_file, sex_encoder_file
        )
        
        # Prepare the input data
        x_new = pd.DataFrame({
            'island': [island],
            'culmen_length_mm': [culmen_length],
            'culmen_depth_mm': [culmen_depth],
            'flipper_length_mm': [flipper_length],
            'body_mass_g': [body_mass],
            'sex': [sex]
        })

        # Apply the transformations using the encoders
        x_new['island'] = island_encoder.transform(x_new['island'])
        x_new['sex'] = sex_encoder.transform(x_new['sex'])

        # Make prediction
        y_pred_new = model.predict(x_new)

        # Decode the prediction to species
        result = species_encoder.inverse_transform(y_pred_new)

        # Display the result
        st.write(f'The predicted species is: {result[0]}')
    else:
        st.warning("Please upload all the required files (model, species encoder, island encoder, and sex encoder).")


