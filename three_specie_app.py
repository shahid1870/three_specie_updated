#maka an app of strem which takes three inputs:
'Present_Surface_Temperature_Max_asc' 'Present_Surface_Salinity_Mean_asc' 'Present_Surface_Salinity_max_asc'
# output teo values: 'latitude_c', 'longitude_c'

# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys
from streamlit_folium import folium_static
import sklearn
import xgboost
# model_path_test = 'voting_halodule_2050_new.pkl'
# with open(model_path_test, 'rb') as file:
#     model_test_2050 = pickle.load(file)
#another way to laod a model
#model_test_2050 = pickle.load(open('voting_halodule_2050_new', 'rb'))

#sider bar if selected Halodule Uninervis the run teh code
st.sidebar.title('Select the species')

#select the species

species = st.sidebar.selectbox('Select the species', ('Halodule Uninervis Current Distrbution','Halodule Uninervis Temporal Distrbution (2050)','Halodule Uninervis Temporal Distrbution (2100)','Halodule Pinifolia Current Distribution','Halodule Pinifolia Temporal Distrbution (2050)',
'Halodule Pinifolia Temporal Distrbution (2100)','Thalassia Hemprichii Current Distribution','Thalassia Hemprichii Temporal Distrbution (2050)','Thalassia Hemprichii Temporal Distrbution (2100)'))


#Halodule Uninervis
if species == 'Halodule Uninervis Current Distrbution':

    #heading
    #write tile with mid align

    st.markdown("<h1 style='text-align: center; color: black;'>AI based Prediction: SDM </h1>", unsafe_allow_html=True)


    #Pciture
    st.image('Halodule_uninervis_new.jpg', caption='Halodule Uninervis', use_column_width=True)


    # Load the pickled model
    model_path = 'best_model_Halodule_Unnivers_current_14_jan_2024.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Define the Streamlit app
    def halodule_Uninervis_current():
        st.title('Sdm: Prediction')
        st.write('Enter the input variables to predict latitude and longitude:')

        # Input variables
        present_surface_temperature_min = st.number_input('Present Surface Temperature Min', min_value=0.0)
        present_surface_temperature_mean = st.number_input('Present Surface Temperature Mean', min_value=0.0)
        present_surface_temperature_max = st.number_input('Present Surface Temperature Max', min_value=0.0)
        present_surface_salinity_mean = st.number_input('Present Surface Salinity Mean', min_value=0.0)


        # Predict button
        if st.button('Predict'):
            # Prepare the input data
            input_data = np.array([[
                present_surface_temperature_max,
                present_surface_temperature_mean,
                present_surface_temperature_max,
                present_surface_salinity_mean
]])


            # Make the prediction
            prediction = model.predict(input_data)

            # Display the predicted latitude and longitude
            st.write('Predicted Latitude: ', prediction[0][0])
            st.write('Predicted Longitude: ', prediction[0][1])
            # Create a map centered at the predicted latitude and longitude
            map_center = [prediction[0][0], prediction[0][1]]
            import folium
            m = folium.Map(location=map_center, zoom_start=12)

            # Add a marker at the predicted latitude and longitude
            folium.Marker(location=map_center).add_to(m)

            # Display the map
            folium_static(m)

# Run the app
    halodule_Uninervis_current()
#Halodule Uninervis
elif species == 'Halodule Uninervis Temporal Distrbution (2050)':

    #heading
    #write tile with mid align

    st.markdown("<h1 style='text-align: center; color: black;'>AI based Prediction: SDM </h1>", unsafe_allow_html=True)


    #Pciture
    st.image('Halodule_uninervis_new.jpg', caption='Halodule Uninervis', use_column_width=True)


    # Load the pickled model
    model_path = 'best_model_Halodule_Unnivers_2050_14_jan_2024.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Define the Streamlit app
    def halodule_Uninervis_2050():
        st.title('Sdm: Prediction')
        st.write('Enter the input variables to predict latitude and longitude:')

        # For RCP26 in 2050
        rcp26_surface_salinity_mean_2050 = st.number_input('2050AOGCM.RCP26.Surface.Salinity.Mean', min_value=0.0)
        rcp26_surface_temperature_max_2050 = st.number_input('2050AOGCM.RCP26.Surface.Temperature.Max', min_value=0.0)
        rcp26_surface_temperature_mean_2050 = st.number_input('2050AOGCM.RCP26.Surface.Temperature.Mean', min_value=0.0)
        rcp26_surface_temperature_min_2050 = st.number_input('2050AOGCM.RCP26.Surface.Temperature.Min', min_value=0.0)

# For RCP85 in 2050
        rcp85_surface_salinity_mean_2050 = st.number_input('2050AOGCM.RCP85.Surface.Salinity.Mean', min_value=0.0)
        rcp85_surface_temperature_max_2050 = st.number_input('2050AOGCM.RCP85.Surface.Temperature.Max', min_value=0.0)
        rcp85_surface_temperature_mean_2050 = st.number_input('2050AOGCM.RCP85.Surface.Temperature.Mean', min_value=0.0)
        rcp85_surface_temperature_min_2050 = st.number_input('2050AOGCM.RCP85.Surface.Temperature.Min', min_value=0.0)


        # Predict button
        if st.button('Predict'):
            # Prepare the input data
            input_data = np.array([[
                rcp26_surface_salinity_mean_2050,
                rcp26_surface_temperature_max_2050,
                rcp26_surface_temperature_mean_2050,
                rcp26_surface_temperature_min_2050,
                rcp85_surface_salinity_mean_2050,
                rcp85_surface_temperature_max_2050,
                rcp85_surface_temperature_mean_2050,
                rcp85_surface_temperature_min_2050
]])


            # Make the prediction
            prediction = model.predict(input_data)

            # Display the predicted latitude and longitude
            st.write('Predicted Latitude: ', prediction[0][0])
            st.write('Predicted Longitude: ', prediction[0][1])
            # Create a map centered at the predicted latitude and longitude
            map_center = [prediction[0][0], prediction[0][1]]
            import folium
            m = folium.Map(location=map_center, zoom_start=12)

            # Add a marker at the predicted latitude and longitude
            folium.Marker(location=map_center).add_to(m)

            # Display the map
            folium_static(m)

# Run the app
    halodule_Uninervis_2050()

#Halodule Uninervis
elif species == 'Halodule Uninervis Temporal Distrbution (2100)':

    #heading
    #write tile with mid align

    st.markdown("<h1 style='text-align: center; color: black;'>AI based Prediction: SDM </h1>", unsafe_allow_html=True)


    #Pciture
    st.image('Halodule_uninervis_new.jpg', caption='Halodule Uninervis', use_column_width=True)


    # Load the pickled model
    model_path = 'best_model_Halodule_Unnivers_2050_14_jan_2024.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Define the Streamlit app
    def halodule_Uninervis_2100():
        st.title('Sdm: Prediction')
        st.write('Enter the input variables to predict latitude and longitude:')

# For RCP26 in 2100
        rcp26_surface_salinity_mean_2100 = st.number_input('2100AOGCM.RCP26.Surface.Salinity.Mean', min_value=0.0)
        rcp26_surface_temperature_max_2100 = st.number_input('2100AOGCM.RCP26.Surface.Temperature.Max', min_value=0.0)
        rcp26_surface_temperature_mean_2100 = st.number_input('2100AOGCM.RCP26.Surface.Temperature.Mean', min_value=0.0)
        rcp26_surface_temperature_min_2100 = st.number_input('2100AOGCM.RCP26.Surface.Temperature.Min', min_value=0.0)

        # For RCP85 in 2100
        rcp85_surface_salinity_mean_2100 = st.number_input('2100AOGCM.RCP85.Surface.Salinity.Mean', min_value=0.0)
        rcp85_surface_temperature_max_2100 = st.number_input('2100AOGCM.RCP85.Surface.Temperature.Max', min_value=0.0)
        rcp85_surface_temperature_mean_2100 = st.number_input('2100AOGCM.RCP85.Surface.Temperature.Mean', min_value=0.0)
        rcp85_surface_temperature_min_2100 = st.number_input('2100AOGCM.RCP85.Surface.Temperature.Min', min_value=0.0)


        # Predict button
        if st.button('Predict'):
            # Prepare the input data
            input_data = np.array([[

                    rcp26_surface_salinity_mean_2100,
                    rcp26_surface_temperature_max_2100,
                    rcp26_surface_temperature_mean_2100,
                    rcp26_surface_temperature_min_2100,
                    rcp85_surface_salinity_mean_2100,
                    rcp85_surface_temperature_max_2100,
                    rcp85_surface_temperature_mean_2100,
                    rcp85_surface_temperature_min_2100
]])


            # Make the prediction
            prediction = model.predict(input_data)

            # Display the predicted latitude and longitude
            st.write('Predicted Latitude: ', prediction[0][0])
            st.write('Predicted Longitude: ', prediction[0][1])
            # Create a map centered at the predicted latitude and longitude
            map_center = [prediction[0][0], prediction[0][1]]
            import folium
            m = folium.Map(location=map_center, zoom_start=12)

            # Add a marker at the predicted latitude and longitude
            folium.Marker(location=map_center).add_to(m)

            # Display the map
            folium_static(m)

# Run the app
    halodule_Uninervis_2100()




elif species == 'Halodule Pinifolia Current Distribution':
    # Write title with mid align
        st.markdown("<h1 style='text-align: center; color: black;'>AI based Prediction: SDM </h1>", unsafe_allow_html=True)

    # Picture
        st.image('Thalassia pinfolia_new.jpg', caption='Halodule Pinifolia', use_column_width=True)

    
        # Load the pickled model
        model_path = 'rf_halodule_current_distribution.pkl'
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        # Define the Streamlit app function
        def halodule1_current():
            st.title('Sdm: Prediction')
            
            # Create a form to organize the input elements
            
            st.write('Enter the input variables to predict latitude and longitude:')
                # Input variables
            present_surface_salinity_mean = st.number_input('Present Surface Salinity Mean', min_value=0.0)
            present_surface_temperature_max = st.number_input('Present Surface Temperature Max', min_value=0.0)
            present_surface_temperature_mean = st.number_input('Present Surface Temperature Mean', min_value=0.0)
            present_surface_temperature_min = st.number_input('Present Surface Temperature Min', min_value=0.0)
            
            

            if st.button('Predict'):
                # Prepare the input data
                input_data = np.array([[present_surface_salinity_mean,
                    present_surface_temperature_max,
                    present_surface_temperature_mean,
                    present_surface_temperature_min
                ]])
            # Make the prediction
                prediction = model.predict(input_data)

                # Display the predicted latitude and longitude
                st.write('Predicted Latitude: ', prediction[0][0])
                st.write('Predicted Longitude: ', prediction[0][1])
                # Create a map centered at the predicted latitude and longitude
                map_center = [prediction[0][0], prediction[0][1]]
                import folium
                m = folium.Map(location=map_center, zoom_start=12)

                # Add a marker at the predicted latitude and longitude
                folium.Marker(location=map_center).add_to(m)

            # Display the map
                folium_static(m)

        # Call the app function
        halodule1_current()



                # # Create a map centered at the predicted latitude and longitude
                # map_center = [prediction[0][0], prediction[0][1]]
                # m = folium.Map(location=map_center, zoom_start=12)

                # # Add a marker at the predicted latitude and longitude
                # folium.Marker(location=map_center).add_to(m)

                # # Display the map
                # folium_static(m)

        # Call the app function


elif species == 'Halodule Pinifolia Temporal Distrbution (2050)':

    
        # Load the pickled model
        st.markdown("<h1 style='text-align: center; color: black;'>AI based Prediction: SDM </h1>", unsafe_allow_html=True)


        #Pciture
        st.image('Thalassia pinfolia_new.jpg', caption='Halodule Pinifolia', use_column_width=True)
        #the file below is not accessable why?

        model_path = 'xgb_halodule_pinifolia_2050.pkl'
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        # Define the Streamlit app
        def halodule_pinfole_2050():
            st.title('Sdm: Prediction')
            st.write('Enter the input variables to predict latitude and longitude:')

            # Input variables
            # 2050AOGCM.RCP26.Surface.Salinity.Mean
            # 2050AOGCM.RCP26.Surface.Temperature.Max
            # 2050AOGCM.RCP26.Surface.Temperature.Mean
            # 2050AOGCM.RCP26.Surface.Temperature.Min
            # 2050AOGCM.RCP85.Surface.Temperature.Max
            # 2050AOGCM.RCP85.Surface.Salinity.Mean

            #make the number_input acording to the above varibles

            AOGCM_RCP26_Surface_Salinity_Mean = st.number_input('2050AOGCM RCP26 Surface Salinity Mean', min_value=0.0)
            AOGCM_RCP26_Surface_Temperature_Max = st.number_input('2050AOGCM.RCP26 Surface Temperature Max', min_value=0.0)
            AOGCM_RCP26_Surface_Temperature_Mean = st.number_input('2050AOGCM RCP26 Surface Temperature Mean', min_value=0.0)
            AOGCM_RCP26_Surface_Temperature_Min = st.number_input('2050AOGCM RCP26 Surface Temperature.Min', min_value=0.0)
            AOGCM_RCP85_Surface_Temperature_Max = st.number_input('2050AOGCM RCP85 Surface Temperature.Max', min_value=0.0)
            AOGCM_RCP85_Surface_Salinity_Mean = st.number_input('2050AOGCM RCP85 Surface Salinity Mean', min_value=0.0)

            #make input data
            input_data = np.array([[
                AOGCM_RCP26_Surface_Salinity_Mean,
                AOGCM_RCP26_Surface_Temperature_Max,
                AOGCM_RCP26_Surface_Temperature_Mean,
                AOGCM_RCP26_Surface_Temperature_Min,
                AOGCM_RCP85_Surface_Temperature_Max,
                AOGCM_RCP85_Surface_Salinity_Mean
            ]])

            #make prediction buton
            if st.button('Predict'):
                 
                # Make the prediction
                prediction = model.predict(input_data)

                    # Display the predicted latitude and longitude
                st.write('Predicted Latitude: ', prediction[0][0])
                st.write('Predicted Longitude: ', prediction[0][1])
                    # Create a map centered at the predicted latitude and longitude
                map_center = [prediction[0][0], prediction[0][1]]
                import folium
                m = folium.Map(location=map_center, zoom_start=12)

                    # Add a marker at the predicted latitude and longitude
                folium.Marker(location=map_center).add_to(m)

                    # Display the map
                folium_static(m)



        halodule_pinfole_2050() 

elif species == 'Halodule Pinifolia Temporal Distrbution (2100)':
        st.markdown("<h1 style='text-align: center; color: black;'>AI based Prediction: SDM </h1>", unsafe_allow_html=True)


        #Pciture
        st.image('Thalassia pinfolia_new.jpg', caption='Halodule Pinifolia', use_column_width=True)
        # Load the pickled model
        model_path = 'xgb_halodule_pinifolia_2100.pkl'
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        # Define the Streamlit app
        def main():
            st.title('Sdm: Prediction')
            st.write('Enter the input variables to predict latitude and longitude:')

            # Input variables
            # 2100AOGCM.RCP26.Surface.Salinity.Mean
            # 2100AOGCM.RCP26.Surface.Temperature.Max
            # 2100AOGCM.RCP26.Surface.Temperature.Min 
            # 2100AOGCM.RCP26.Surface.Temperature.Mean     
            # 2100AOGCM.RCP26.Surface.Temperature.Max2     
            # 2100AOGCM.RCP85.Surface.Temperature.Mean     
            # 2100AOGCM.RCP85.Surface.Salinity.Mean        
            # 2100AOGCM.RCP85.Surface.Salinity.Mean2       
            # 2100AOGCM.RCP85.Surface.Temperature.Mean3    
            # 2100AOGCM.RCP85.Surface.Temperature.Min      

            #make the number_input acording to the above varibles

            AOGCM_RCP26_Surface_Salinity_Mean = st.number_input('2100AOGCM RCP26 Surface Salinity Mean', min_value=0.0)
            AOGCM_RCP26_Surface_Temperature_Max = st.number_input('2100AOGCM RCP26 Surface Temperature Max', min_value=0.0)
            AOGCM_RCP26_Surface_Temperature_Min = st.number_input('2100AOGCM RCP26 Surface Temperature Min', min_value=0.0)
            AOGCM_RCP26_Surface_Temperature_Mean = st.number_input('2100AOGCM RCP26 Surface Temperature Mean', min_value=0.0)
            AOGCM_RCP26_Surface_Temperature_Max2 = st.number_input('2100AOGCM RCP26 Surface Temperature Max2', min_value=0.0)
            AOGCM_RCP85_Surface_Temperature_Mean = st.number_input('2100AOGCM RCP85 Surface Temperature Mean', min_value=0.0)
            AOGCM_RCP85_Surface_Salinity_Mean = st.number_input('2100AOGCM RCP85 Surface Salinity Mean', min_value=0.0)
            AOGCM_RCP85_Surface_Salinity_Mean2 = st.number_input('2100AOGCM RCP85 Surface Salinity Mean2', min_value=0.0)
            AOGCM_RCP85_Surface_Temperature_Mean3 = st.number_input('2100AOGCM RCP85 Surface Temperature Mean3', min_value=0.0)
            AOGCM_RCP85_Surface_Temperature_Min = st.number_input('2100AOGCM RCP85 Surface Temperature Min', min_value=0.0)

            #make input data
            input_data = np.array([[
                AOGCM_RCP26_Surface_Salinity_Mean,
                AOGCM_RCP26_Surface_Temperature_Max,
                AOGCM_RCP26_Surface_Temperature_Min,
                AOGCM_RCP26_Surface_Temperature_Mean,
                AOGCM_RCP26_Surface_Temperature_Max2,
                AOGCM_RCP85_Surface_Temperature_Mean,
                AOGCM_RCP85_Surface_Salinity_Mean,
                AOGCM_RCP85_Surface_Salinity_Mean2,
                AOGCM_RCP85_Surface_Temperature_Mean3,
                AOGCM_RCP85_Surface_Temperature_Min
            ]])
            #make prediction buton
            if st.button('Predict'):
                 
                # Make the prediction
                prediction = model.predict(input_data)

                    # Display the predicted latitude and longitude
                st.write('Predicted Latitude: ', prediction[0][0])
                st.write('Predicted Longitude: ', prediction[0][1])
                    # Create a map centered at the predicted latitude and longitude
                map_center = [prediction[0][0], prediction[0][1]]
                import folium
                m = folium.Map(location=map_center, zoom_start=12)

                    # Add a marker at the predicted latitude and longitude
                folium.Marker(location=map_center).add_to(m)

                    # Display the map
                folium_static(m)



        # Run the app
        if __name__ == '__main__':
            main() 

#Thalassia
elif species == 'Thalassia Hemprichii Current Distribution':
    #heading
    #write tile with mid align

        st.markdown("<h1 style='text-align: center; color: black;'>AI based Prediction: SDM </h1>", unsafe_allow_html=True)


    #Pciture
        st.image('Thalassia hemprichii_new.jpg', caption='Thalassia Hemprichii', use_column_width=True)

        # Load the pickled model
        model_path = 'voting_thalassia_current_distribution'
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        # Define the Streamlit app
        def main():
            st.title('Sdm: Prediction')
            st.write('Enter the input variables to predict latitude and longitude:')

            # Input variables
            # Present_surface_temperature_min             
            # Present_surface_temperature_mean            
            # Present_surface_temperature_max             
            # Present_surface_salinity   

            #make the number_input acording to the above varibles

            Present_surface_temperature_min = st.number_input('Present surface temperature min', min_value=0.0)
            Present_surface_temperature_mean = st.number_input('Present surface temperature mean', min_value=0.0)
            Present_surface_temperature_max = st.number_input('Present surface temperature max', min_value=0.0)
            Present_surface_salinity = st.number_input('Present surface salinity', min_value=0.0)
            #make input data
            input_data = np.array([[
                Present_surface_temperature_min,
                Present_surface_temperature_mean,
                Present_surface_temperature_max,
                Present_surface_salinity
            ]])
            #make prediction buton
            if st.button('Predict'):
                 
                  # Make the prediction
                prediction = model.predict(input_data)

                    # Display the predicted latitude and longitude
                st.write('Predicted Latitude: ', prediction[0][0])
                st.write('Predicted Longitude: ', prediction[0][1])
                    # Create a map centered at the predicted latitude and longitude
                map_center = [prediction[0][0], prediction[0][1]]
                import folium
                m = folium.Map(location=map_center, zoom_start=12)

                    # Add a marker at the predicted latitude and longitude
                folium.Marker(location=map_center).add_to(m)

                    # Display the map
                folium_static(m)



        # Run the app
        if __name__ == '__main__':
            main()

elif species == 'Thalassia Hemprichii Temporal Distrbution (2050)':
        st.markdown("<h1 style='text-align: center; color: black;'>AI based Prediction: SDM </h1>", unsafe_allow_html=True)


    #Pciture
        st.image('Thalassia hemprichii_new.jpg', caption='Thalassia Hemprichii', use_column_width=True)
        # Load the pickled model
        model_path = 'xgb_thalassia_2050'
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        # Define the Streamlit app
        def main():
            st.title('Sdm: Prediction')
            st.write('Enter the input variables to predict latitude and longitude:')

            # Input variables
            # 2050AOGCM.RCP26.Surface.Salinity.Mean       0
            # 2050AOGCM.RCP26.Surface.Temperature.Max     0
            # 2050AOGCM.RCP26.Surface.Temperature.Mean    0
            # 2050AOGCM.RCP26.Surface.Temperature.Min     0
            # 2050AOGCM.RCP85.Surface.Salinity.Mean.      0
            # 2050AOGCM.RCP85.Surface.Temperature.Max     0
            # 2050AOGCM.RCP85.Surface.Temperature.Mean    0
            # 2050AOGCM.RCP85.Surface.Temperature.Min     0

            #make the number_input acording to the above varibles

            AOGCM_RCP26_Surface_Salinity_Mean = st.number_input('2050AOGCM RCP26 Surface Salinity Mean', min_value=0.0)
            AOGCM_RCP26_Surface_Temperature_Max = st.number_input('2050AOGCM RCP26 Surface Temperature Max', min_value=0.0)
            AOGCM_RCP26_Surface_Temperature_Mean = st.number_input('2050AOGCM RCP26 Surface Temperature Mean', min_value=0.0)
            AOGCM_RCP26_Surface_Temperature_Min = st.number_input('2050AOGCM RCP26 Surface Temperature Min', min_value=0.0)
            AOGCM_RCP85_Surface_Salinity_Mean = st.number_input('2050AOGCM RCP85 Surface Salinity Mean', min_value=0.0)
            AOGCM_RCP85_Surface_Temperature_Max = st.number_input('2050AOGCM RCP85 Surface Temperature Max', min_value=0.0)
            AOGCM_RCP85_Surface_Temperature_Mean = st.number_input('2050AOGCM RCP85 Surface Temperature Mean', min_value=0.0)
            AOGCM_RCP85_Surface_Temperature_Min = st.number_input('2050AOGCM RCP85 Surface Temperature Min', min_value=0.0)

            #make input data
            input_data = np.array([[
                AOGCM_RCP26_Surface_Salinity_Mean,
                AOGCM_RCP26_Surface_Temperature_Max,
                AOGCM_RCP26_Surface_Temperature_Mean,
                AOGCM_RCP26_Surface_Temperature_Min,
                AOGCM_RCP85_Surface_Salinity_Mean,
                AOGCM_RCP85_Surface_Temperature_Max,
                AOGCM_RCP85_Surface_Temperature_Mean,
                AOGCM_RCP85_Surface_Temperature_Min
            ]])


            #make prediction buton
            if st.button('Predict'):
                 
                # Make the prediction
                prediction = model.predict(input_data)

                    # Display the predicted latitude and longitude
                st.write('Predicted Latitude: ', prediction[0][0])
                st.write('Predicted Longitude: ', prediction[0][1])
                    # Create a map centered at the predicted latitude and longitude
                map_center = [prediction[0][0], prediction[0][1]]
                import folium
                m = folium.Map(location=map_center, zoom_start=12)

                    # Add a marker at the predicted latitude and longitude
                folium.Marker(location=map_center).add_to(m)

                    # Display the map
                folium_static(m)



        # Run the app
        if __name__ == '__main__':
            main() 

elif species == 'Thalassia Hemprichii Temporal Distrbution (2100)':
        st.markdown("<h1 style='text-align: center; color: black;'>AI based Prediction: SDM </h1>", unsafe_allow_html=True)


    #Pciture
        st.image('Thalassia hemprichii_new.jpg', caption='Thalassia Hemprichii', use_column_width=True)
        # Load the pickled model
        model_path = 'xgb_thalassia_2100'
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        # Define the Streamlit app
        def main():
            st.title('Sdm: Prediction')
            st.write('Enter the input variables to predict latitude and longitude:')

            # Input variables
            # 2100AOGCM.RCP85.Surface.Salinity.Mean.      
            # 2100AOGCM.RCP85.Surface.Temperature.Max     
            # 2100AOGCM.RCP85.Surface.Temperature.Mean    
            # 2100AOGCM.RCP85.Surface.Temperature.Min     
            # 2100AOGCM.RCP26.Surface.Temperature.Max     
            # 2100AOGCM.RCP26.Surface.Temperature.Mean    
            # 2100AOGCM.RCP26.Surface.Temperature.Min     
            # 2100AOGCM.RCP26.Surface.Salinity.Mean             

            #make the number_input acording to the above varibles

            AOGCM_RCP85_Surface_Salinity_Mean = st.number_input('2100AOGCM RCP85 Surface Salinity Mean', min_value=0.0)
            AOGCM_RCP85_Surface_Temperature_Max = st.number_input('2100AOGCM RCP85 Surface Temperature Max', min_value=0.0)
            AOGCM_RCP85_Surface_Temperature_Mean = st.number_input('2100AOGCM RCP85 Surface Temperature Mean', min_value=0.0)
            AOGCM_RCP85_Surface_Temperature_Min = st.number_input('2100AOGCM RCP85 Surface Temperature Min', min_value=0.0)
            AOGCM_RCP26_Surface_Temperature_Max = st.number_input('2100AOGCM RCP26 Surface Temperature Max', min_value=0.0)
            AOGCM_RCP26_Surface_Temperature_Mean = st.number_input('2100AOGCM RCP26 SurfaceTemperature Mean', min_value=0.0)
            AOGCM_RCP26_Surface_Temperature_Min = st.number_input('2100AOGCM RCP26 SurfaceTemperature Min', min_value=0.0)
            AOGCM_RCP26_Surface_Salinity_Mean = st.number_input('2100AOGCM RCP26 Surface Salinity Mean', min_value=0.0)
            
            #make input data
            input_data = np.array([[
                AOGCM_RCP85_Surface_Salinity_Mean,
                AOGCM_RCP85_Surface_Temperature_Max,
                AOGCM_RCP85_Surface_Temperature_Mean,
                AOGCM_RCP85_Surface_Temperature_Min,
                AOGCM_RCP26_Surface_Temperature_Max,
                AOGCM_RCP26_Surface_Temperature_Mean,
                AOGCM_RCP26_Surface_Temperature_Min,
                AOGCM_RCP26_Surface_Salinity_Mean
            ]])
            #make prediction buton
            if st.button('Predict'):
                # Make the prediction
                prediction = model.predict(input_data)

                    # Display the predicted latitude and longitude
                st.write('Predicted Latitude: ', prediction[0][0])
                st.write('Predicted Longitude: ', prediction[0][1])
                    # Create a map centered at the predicted latitude and longitude
                map_center = [prediction[0][0], prediction[0][1]]
                import folium
                m = folium.Map(location=map_center, zoom_start=12)

                    # Add a marker at the predicted latitude and longitude
                folium.Marker(location=map_center).add_to(m)

                    # Display the map
                folium_static(m)



        # Run the app
        if __name__ == '__main__':
            main() 


