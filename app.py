
import streamlit as st
import pandas as pd 
import numpy as np 
import plotly.express as px
import seaborn as sns 
import pickle
import warnings
warnings.filterwarnings("ignore")

def main():
    st.title("Bank Deposit Prediction System")
    
    data = pd.read_csv("bank.csv")
    st.write("Shape of DataSet", data.shape)

    # Sidebar Styling
    st.sidebar.markdown("<h2 style='text-align: center; color: green;'>\U0001F3E6 Bank Deposit Prediction \U0001F3E6</h2>", unsafe_allow_html=True)
    st.sidebar.image("https://shorturl.at/S2n5D", width=500)
    menu = st.sidebar.radio("Menu", ["📊 Analysis", "🔮 Prediction"])
    st.sidebar.markdown("---")  
    st.sidebar.info("💡 **Tip:** Make smarter financial decisions with AI!")
    
    if menu == "📊 Analysis":
        st.image("bank.jpg", width=550)
        st.header("Tabular Data of Bank")
        if st.checkbox("Tabular Data"):
            st.table(data.head(150))
        st.header("Statistical Summary of Dataset")
        if st.checkbox("Statistics"):
            st.table(data.describe())
        
        st.title("Graphs")
        graph = st.selectbox("Different Types of Graphs", ["Line Graph", "Scatter Plot"])
        
        if graph == "Line Graph":
            monthly_deposits = data.groupby('month')['deposit'].value_counts().unstack().fillna(0)
            fig = px.line(monthly_deposits, x=monthly_deposits.index, y=monthly_deposits.columns,
                          title="Monthly Deposit Trend", labels={'index': 'Month', 'value': 'Number of Customers'})
            st.plotly_chart(fig)
        
        if graph == "Scatter Plot":
            age_val = st.slider("Filter Data Using Age:", 18, 95)
            data_filtered = data[data["age"] >= age_val]
            fig = px.scatter(data_filtered, x="balance", y="age", color="deposit", title="Age Vs Bank Balance")
            st.plotly_chart(fig)
        
        st.header("Correlation Graph")
        corr_matrix = data.corr()
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='rdylbu')
        st.plotly_chart(fig)
    
    if menu == "🔮 Prediction":
        with open('model.pkl', 'rb') as pkl:
            classifier = pickle.load(pkl)
        with open("scaler_model.pkl", 'rb') as pkl1:
            sc = pickle.load(pkl1)
        
        left, right = st.columns((2, 2))
        inputs = {
            'age': left.number_input('Enter Age', step=1, value=0),
            'job': right.number_input('Job Level (1-10)', step=1, value=0),
            'marital': left.number_input('Marital Status (0-2)', step=1, value=0),
            'education': right.number_input('Education Level (1-3)', step=1, value=0),
            'default': left.number_input('Credit Card Default (0-No,1-Yes)', step=1, value=0),
            'balance': right.number_input('Enter Bank Balance', step=1, value=0),
            'housing': left.number_input('Housing Loan (0-No,1-Yes)', step=1, value=0),
            'loan': right.number_input('Personal Loan (0-No,1-Yes)', step=1, value=0),
            'contact': left.number_input('Contact Level (0-2)', step=1, value=0),
            'day': right.number_input('Enter Days (0-31)', step=1, value=0),
            'month': left.number_input('Enter Month (0-11)', step=1, value=0),
            'duration': right.number_input('Enter Duration', step=1, value=0),
            'campaign': left.number_input('Enter Campaign Count', step=1, value=0),
            'pdays': right.number_input('Enter Past Days', step=1, value=0),
            'previous': left.number_input('Enter Previous Contacts', step=1, value=0),
            'poutcome': right.number_input('Outcome (0-3)', step=1, value=0)
        }
        
        predict_button = st.button('Will Make Deposit?')
        
        if predict_button:
            input_data = np.array([list(inputs.values())]).reshape(1, -1)
            std_data = sc.transform(input_data)
            res = classifier.predict(std_data)
            
            if res[0] == 1:
                st.success("Will Make Deposit")
                st.image("https://shorturl.at/Cp7vo", width=1000)
                st.toast("Deposit Successful! 💰")
            else:
                st.success("Will Not Make Deposit")

main()

