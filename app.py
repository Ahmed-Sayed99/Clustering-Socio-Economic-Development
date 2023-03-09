import pandas as pd
import streamlit as st
import joblib

st.title('Model Deployment: World Development')

st.sidebar.header('User Input Parameters')

# load the model from disk
scaling = joblib.load('scaler.pkl')
features = joblib.load('pca.pkl')
model = joblib.load('kmeans.pkl')


def user_input_features():
    Birth_Rate = st.sidebar.number_input("Insert Birth Rate")
    Business_Tax_Rate = st.sidebar.number_input("Insert Business Tax Rate")
    CO2_Emissions = st.sidebar.number_input("Insert CO2 emissions")
    Days_to_Start_Business = st.sidebar.number_input("Insert Days to Start Business")
    Energy_Usage = st.sidebar.number_input("Insert Energy Usage")
    GDP = st.sidebar.number_input("Insert GDP_$")
    Health_Exp_in_pc_GDP = st.sidebar.number_input("Health Exp % GDP")
    Health_Exp_per_Capita = st.sidebar.number_input("Insert Health Exp/Capita_$")
    Hours_to_do_Tax = st.sidebar.number_input("Insert Hours to do Tax")
    IMR = st.sidebar.number_input("Insert Infant Mortality Rate")
    Internet_Usage = st.sidebar.number_input("InsertInternet Usage")
    Lending_Interest = st.sidebar.number_input("Insert Lending Interest")
    Life_Expectancy_Female = st.sidebar.number_input("Insert Life Expectancy Female")
    Life_Expectancy_Male = st.sidebar.number_input("Life Expectancy Male")
    Mobile_Phone_Usage = st.sidebar.number_input("Insert Mobile Phone Usage")
    Number_of_Records = st.sidebar.number_input("Insert Number of Records")
    Population_0_14 = st.sidebar.number_input("Population 0-14")
    Population_15_64 = st.sidebar.number_input("Insert Population 15-64")
    Population_65 = st.sidebar.number_input("Population 65")
    Population_Total = st.sidebar.number_input("Insert Population Total")
    Population_Urban = st.sidebar.number_input("Insert Population Urban")
    Tourism_Inbound = st.sidebar.number_input("Insert Tourism Inbound_$")
    Tourism_Outbound = st.sidebar.number_input("Insert Tourism Outbound_$")

    data = {'Birth Rate': Birth_Rate,
            'Business Tax Rate': Business_Tax_Rate,
            'CO2 Emissions': CO2_Emissions,
            'Days to Start Business': Days_to_Start_Business,
            'Energy Usage': Energy_Usage,
            'GDP': GDP,
            'Health Exp % GDP': Health_Exp_in_pc_GDP,
            'Health Exp/Capita': Health_Exp_per_Capita,
            'Hours to do Tax': Hours_to_do_Tax,
            'Infant Mortality Rate': IMR,
            'Internet Usage': Internet_Usage,
            'Lending Interest': Lending_Interest,
            'Life Expectancy Female': Life_Expectancy_Female,
            'Life Expectancy Male': Life_Expectancy_Male,
            'Mobile Phone Usage': Mobile_Phone_Usage,
            'Number of Records': Number_of_Records,
            'Population 0-14': Population_0_14,
            'Population 15-64': Population_15_64,
            'Population 65+': Population_65,
            'Population Total': Population_Total,
            'Population Urban': Population_Urban,
            'Tourism Inbound': Tourism_Inbound,
            'Tourism Outbound': Tourism_Outbound,
            }
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

new_data_norm = scaling.transform(df)
new_data_pca = features.transform(new_data_norm)
prediction = model.predict(new_data_pca)

st.subheader('Predicted Result')

st.write('ClusterID', prediction)

if prediction == 0:
    string = 'Under-Developed'
elif prediction == 1:
    string = 'Developed'
elif prediction == 2:
    string = 'Developing'

st.write('Country Development:', string)

st.write(pd.DataFrame({
    'CLusterID': [0, 1, 2],
    'Development': ['Under-Developed', 'Developed',
                    'Developing'],
}))


