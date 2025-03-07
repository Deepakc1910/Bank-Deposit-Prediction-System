import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import pickle
import warnings
warnings.filterwarnings("ignore")


def main():
    st.title("Bank Deposit Prediction System")


    data=pd.read_csv("bank.csv")
    st.write("Shape of DataSet",data.shape)


    # 🎨 Sidebar Styling
    st.sidebar.markdown("<h2 style='text-align: center; color: green;'>🏦 Bank Deposit Prediction 🏦</h2>", unsafe_allow_html=True)
    
    # 🖼️ Sidebar Image/GIF
    st.sidebar.image("https://shorturl.at/S2n5D", width=500)  # Money GIF

    # 📌 Sidebar Menu with Icons
    menu = st.sidebar.radio("Menu", ["📊 Analysis", "🔮 Prediction"])

    # 🔹 Custom Divider
    st.sidebar.markdown("---")  

    # 💡 Sidebar Info
    st.sidebar.info("💡 **Tip:** Make smarter financial decisions with AI!")

    
    
    if menu=="📊 Analysis":
        st.image("bank.jpg",width=550)
        st.header("Tabular Data of Bank")
        if st.checkbox("Tabular Data"):
            st.table(data.head(150))
        st.header("Statistical Summary of Dataset")
        if st.checkbox("Statistics"):
            st.table(data.describe())
        
        st.title("Graphs")
        graph=st.selectbox("Different Types of Graphs",["Line Graph","Scatter Plot"])
        if graph=="Line Graph":
            monthly_deposits = data.groupby('month')['deposit'].value_counts().unstack().fillna(0)
            fig, ax = plt.subplots(figsize=(10, 6))
            monthly_deposits.plot(kind='line', ax=ax)

            ax.set_title("Monthly Deposit Trend", fontsize=14)
            ax.set_xlabel("Month", fontsize=12)
            ax.set_ylabel("Number of Customers", fontsize=12)
            st.pyplot(fig)
        if graph=="Scatter Plot":
            age_val=st.slider("Filter Data Using Age:",18,95)
            data=data.loc[data["age"]>=age_val]
            fig,ax=plt.subplots(figsize=(10,5))
            sns.scatterplot(data=data,x="balance",y="age",hue="deposit")
            ax.set_title("Age Vs Bank Balance", fontsize=14)
            ax.set_xlabel("Age", fontsize=12)
            ax.set_ylabel("Bank Balance", fontsize=12)
            st.pyplot(fig)
        
        st.header("Correlation Graph")
        fig,ax=plt.subplots(figsize=(12,12))
        sns.heatmap(data.corr(),annot=True,cmap="coolwarm")
        st.pyplot(fig)
        
    if menu=="🔮 Prediction":
        with open('model.pkl','rb') as pkl:
            classifier = pickle.load(pkl)


        with open("scaler_model.pkl",'rb') as pkl1:
            sc=pickle.load(pkl1)
    
        left,right = st.columns((2,2))
        age       = left.number_input('Enter Age as whole numbers', step = 1, value=0,key='age')
        job       = right.number_input('Job (Enter between 1 to 10 as per level)', step = 1, value=0,key='job')
        marital   = left.number_input('Enter Marital Status(0-Unmarried,1-Married,2-Divorced)', step = 1, value=0,key='marital')
        education = right.number_input('Education (Enter between 1 to 3 as per education level)', step = 1, value=0,key='education')
        default   = left.number_input('Defaulter for Credit card Use(0-No,1-yes)', step = 1, value=0,key='default')
        balance   = right.number_input('Enter Bank Balance', step = 1,value=0,key='balance')
        housing   = left.number_input('Housing loan(0-No,1-Yes)', step = 1, value=0,key='housing')
        loan      = right.number_input('Personal loan(0-No,1-Yes)', step = 1, value=0,key='loan')
        contact   = left.number_input('Contact level with bank between 0 to 2', step = 1, value=0,key='contact')
        day       = right.number_input('Enter Days From 0 to 31', step = 1, value=0,key='day')
        month     = left.number_input('Enter Month From 0 to 11', step = 1, value=0,key='month')
        duration  = right.number_input('Enter Duration as whole numbers', step = 1, value=0,key='duration')
        campaign  = left.number_input('Enter campaign as whole numbers', step = 1, value=0,key='campaign')
        pdays     = right.number_input('Enter past days as whole numbers', step = 1,value=0,key='pdays')
        previous  = left.number_input('Enter previous days as whole numbers', step = 1, value=0,key='previous')
        poutcome  = right.number_input('Enter outcome between 0 to 3', step = 1, value=0,key='poutcome')
        predict_button =st.button('Will Make Deposit?')
        
        std_data=sc.transform([[age,job,marital,education,default,balance,housing,loan,contact,day,month,duration,campaign,pdays,previous,poutcome]])
        #When predict button is clicked
        if predict_button:


            res = classifier.predict(std_data)
            if res[0]==1:
                st.success("Will Make Deposit")
                st.image("https://shorturl.at/Cp7vo",width=1000)
                st.toast("Deposit Successful! 💰")

            else:
                st.success("Will Not Make Deposit")
                

            
main()

