import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import plotly.graph_objs as go
import streamlit as st
import plotly.express as px
import random
import datetime
import holidays as hd

us_holidays = hd.US()
holidays = pd.DataFrame(
    {
        "holiday": "holiday",
        "ds": list(us_holidays.keys()),
        "lower_window": 0,
        "upper_window": 0,
    }
)


st.set_page_config(
    page_title="Dashboard",
    page_icon=":chart_with_upwards_trend:",
)
st.title("Cleaning Business Forecast Dashboard")

st.markdown("-----")
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
    date_rng = pd.date_range(start="10/1/2022", end="12/31/2022", freq="3D")

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
    m = Prophet(holidays=holidays)
    m.fit(prophet_df)

    # Make predictions for the next 90 days
    future = m.make_future_dataframe(periods=90, include_history=False)
    forecast = m.predict(future)
    show = forecast[["ds", "yhat"]]
    show.columns = ["Date", "prediction"]
    st.dataframe(show)

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
        width=800,
        height=600,
        mapbox_style="open-street-map",
    )
    # map_fig.update_layout()
    map_fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(map_fig)
with tab2:
    # Generate date range from October 2022 to December 2022
    date_rng = pd.date_range(start="10/1/2022", end="12/31/2022", freq="3D")

    # Create a pandas dataframe with dates and base amount
    df = pd.DataFrame(date_rng, columns=["date"])
    df["base_amount"] = 3000

    # Simulate time series data with normal and exponential growth combined
    np.random.seed(42)
    noise = np.random.normal(0, 40, size=len(date_rng))
    growth = np.exp(np.linspace(0, 0.4, len(date_rng)))
    numbers = np.random.randint(low=8, high=25, size=len(date_rng))
    ts_data = df["base_amount"].values * numbers * growth + noise
    df["amount"] = ts_data

    # Train a Prophet model on the time series data
    prophet_df = df[["date", "amount"]].rename(columns={"date": "ds", "amount": "y"})
    m = Prophet(holidays=holidays)
    m.fit(prophet_df)

    # Make predictions for the next 90 days
    future = m.make_future_dataframe(periods=90, include_history=True)
    forecast = m.predict(future)
    show = forecast[["ds", "yhat"]]
    show.columns = ["Date", "prediction"]
    st.dataframe(show)

    # Plot the forecast with interactive plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast"))
    fig.update_layout(
        title="Projected Revenue for the next three months<br><sup>Commercial</sup>",
        xaxis_title="Date",
        yaxis_title="Revenue",
        title_x=0.3,
    )
    st.plotly_chart(fig, use_container_width=True)
    map_data = pd.read_csv("commercial locations.csv")

    map_fig2 = px.scatter_mapbox(
        map_data,
        lat="location/lat",
        lon="location/lng",
        color="type",
        hover_name="title",
        hover_data=["phone", "website"],
        title="Nursing facilities",
        width=800,
        height=600,
        mapbox_style="open-street-map",
    )
    # map_fig.update_layout()
    map_fig2.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(map_fig2)


with tab3:

    # set the start and end dates
    start_date = "2022-10-01"
    end_date = "2022-12-30"

    # Create a range of dates
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    np.random.seed(42)
    sales = np.exp(np.linspace(0, 3, num=len(dates)))

    # add some random noise to the sales
    sales = [int(s + random.uniform(-5, 5)) for s in sales]

    # cap the sales at a maximum of 35 and a minimum of 16
    sales = [min(max(s, 16), 35) for s in sales]
    sales_data = pd.DataFrame({"ds": dates, "y": sales})

    # create a Prophet model and fit it to the sales data
    model = Prophet(holidays=holidays)
    model.fit(sales_data)

    # make predictions for the next 90 days
    future = model.make_future_dataframe(periods=90, include_history=False)
    forecast = model.predict(future)
    forecast["yhat"] = round(forecast["yhat"])
    show = forecast[["ds", "yhat"]]
    show.columns = ["Date", "prediction"]
    st.dataframe(show)
    # Plot the forecast with interactive plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast"))
    fig.update_layout(
        title="Projected Number of sales for the next three months<br><sup></sup>",
        xaxis_title="Date",
        yaxis_title="Sales",
        title_x=0.3,
    )
    st.plotly_chart(fig, use_container_width=True)

# add a marker symbol to the plot at the specified location


# add a marker symbol to the plot at the specified location
