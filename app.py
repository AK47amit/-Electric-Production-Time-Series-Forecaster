import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 1. App Title & Sidebar
st.set_page_config(page_title="Electricity Forecaster", layout="wide")
st.title("⚡ Electric Production Time Series Forecaster")
st.sidebar.header("User Input")
months = st.sidebar.slider("Forecast Months", 1, 24, 12)

# 2. Load Data
@st.cache_data # Isse app baar-baar load nahi hogi, fast chalegi
def load_data():
    data = pd.read_csv("Electric_Production.csv", index_col='DATE', parse_dates=True)
    return data

df = load_data()

# 3. Layout: Raw Data & Main Graph
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Recent Data")
    st.write(df.tail(10))

with col2:
    st.subheader("Production Trend")
    st.line_chart(df)

# 4. Model & Prediction
st.divider()
st.subheader(f"Next {months} Months Prediction")

if st.button('Run Forecast'):
    with st.spinner('Model training in progress...'):
        # Yahan apna wahi model parameters use karein jo notebook mein kiye the
        # Order (p,d,q) aur Seasonal Order (P,D,Q,s)
        model = SARIMAX(df.iloc[:,0], order=(1,1,1), seasonal_order=(1,1,1,12))
        model_fit = model.fit(disp=False)
        
        forecast = model_fit.forecast(steps=months)
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index[-48:], df.iloc[-48:, 0], label='Past 4 Years')
        ax.plot(forecast.index, forecast, label='Predicted', color='red', marker='o')
        ax.set_title("Future Production Forecast")
        ax.legend()
        st.pyplot(fig)
        
        # Download results
        st.download_button("Download Forecast CSV", forecast.to_csv(), "forecast.csv")