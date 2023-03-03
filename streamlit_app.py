import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import plotly.graph_objs as go
import streamlit as st
import plotly.express as px

st.title("Forecast Dashboard")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["Residential", "Commerical", "Product Sale"])

with tab1:
    # Generate date range from October 2022 to December 2022
    date_rng = pd.date_range(start="10/1/2022", end="12/31/2022", freq="D")

    # Create a pandas dataframe with dates and base amount
    df = pd.DataFrame(date_rng, columns=["date"])
    df["base_amount"] = 170

    # Simulate time series data with normal and exponential growth combined
    np.random.seed(42)
    noise = np.random.normal(0, 40, size=len(date_rng))
    growth = np.exp(np.linspace(0, 0.4, len(date_rng)))
    numbers = np.random.randint(low=8, high=25, size=len(date_rng))
    ts_data = df["base_amount"].values * numbers * growth + noise
    df["amount"] = ts_data

    # Train a Prophet model on the time series data
    prophet_df = df[["date", "amount"]].rename(columns={"date": "ds", "amount": "y"})
    m = Prophet()
    m.fit(prophet_df)

    # Make predictions for the next 90 days
    future = m.make_future_dataframe(periods=90, include_history=False)
    forecast = m.predict(future)

    # Plot the forecast with interactive plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast"))
    fig.update_layout(
        title="Projected Revenue for the next three months<br><sup>Residential</sup>",
        xaxis_title="Date",
        yaxis_title="Revenue",
        title_x=0.3,
    )
    st.plotly_chart(fig, use_container_width=True)
    map_data = pd.read_csv("nursing locations.csv")
    map_fig = px.scatter_mapbox(
        map_data,
        lat="location/lat",
        lon="location/lng",
        hover_name="title",
        hover_data=["phone", "website"],
        title="Nursing facilities",
    )

    map_fig.update_layout(mapbox_style="open-street-map")
    map_fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(map_fig)
