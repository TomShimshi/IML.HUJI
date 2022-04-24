import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = pd.read_csv(filename, parse_dates=[2]).dropna()
    data = data.drop_duplicates()

    # Checks if a line is valid:
    data = data[(data["Day"].isin(range(1, 32)) & data["Month"].isin([1,3,5,7,8,10,12])) |
    (data["Day"].isin(range(1, 31)) & data["Month"].isin([4,6,9,11])) |
    (data["Day"].isin(range(1, 30)) & data["Month"].isin([2]))]

    # Delete unlogical row
    data = data[data["Temp"] > -50]

    data["DayOfYear"] = ((data["Date"]).dt.day_of_year)

    return data


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    observations = load_data("../datasets/City_Temperature.csv")


    # Question 2 - Exploring data for specific country
    # Question A:

    israel_obs = observations[observations["Country"] == "Israel"].copy()

    israel_obs['Year'] = israel_obs['Year'].astype(str)
    fig1 = px.scatter(israel_obs, x="DayOfYear", y="Temp", color="Year")
    fig1.update_layout(xaxis_title="Day Of The Year",
                       yaxis_title="Temperature",
                       title="The Temperature in Israel As A Function Of The Day Of The Year",
                       font=dict(family="Arial", size=18))
    fig1.show()

    # Question B:
    month_temp = israel_obs.groupby('Month').agg('std')
    months = list(range(1, 13))


    fig2 = px.bar(month_temp, x=months, y="Temp")
    fig2.update_layout(xaxis_title="Month",
                      yaxis_title="Temperature",
                      title="The STD Of The Temperature in Israel As A Function Of The Month",
                      font=dict(family="Arial", size=18))
    fig2.show()

    # Question 3 - Exploring differences between countries
    country_temp = observations.groupby(['Country','Month']).agg({"Temp": ['mean', 'std']})
    countries = ["Israel", "Jordan", "South Africa", "The Netherlands"]
    fig3 = go.Figure()

    for i in range(1, 5):
        mean_arr = list(country_temp[("Temp", "mean")][12*(i-1):12*i,])
        std_arr = country_temp[("Temp", "std")][12*(i-1):12*i,]
        fig3.add_trace(go.Scatter(x=months, y=mean_arr, name=countries[i-1],
                                  error_y=dict( type='data',array=std_arr,visible=True)))

    fig3.update_layout(xaxis_title="Month",
                       yaxis_title="Temperature",
                       title="The Temperature in Each Country As A Function Of The Month",
                       font=dict(family="Arial", size=18))
    fig3.show()


    # Question 4 - Fitting model for different values of `k`
    israel_obs = israel_obs.sample(frac=1)
    test_count = int(np.round(0.75 * israel_obs.shape[0]))
    train_set, test_set = israel_obs["DayOfYear"][:test_count], israel_obs["DayOfYear"][
                                                                     test_count:]
    train_y, test_y = israel_obs["Temp"][:test_count], israel_obs["Temp"][
                                                                     test_count:]

    degree = list(range(1,11))
    losses = []
    for k in degree:
        poli_fit = PolynomialFitting(k)
        poli_fit._fit(train_set.to_numpy(), train_y.to_numpy())
        loss = np.round(poli_fit._loss(test_set.to_numpy(), test_y.to_numpy()), decimals=2)
        print("For the degree {0}, the loss recorded is- {1}".format(k, loss))
        losses.append(loss)

    fig4 = px.bar(x=degree, y=losses)
    fig4.update_layout(xaxis_title="K",
                       yaxis_title="Loss",
                       title="The Loss Recorded As A Function Of The Degree",
                       font=dict(family="Arial", size=18))

    fig4.show()

    # Question 5 - Evaluating fitted model on different countries
    K = 5  # Best K from Q-4
    poli_fit_2 = PolynomialFitting(K)
    poli_fit_2._fit(israel_obs["DayOfYear"], israel_obs["Temp"])
    countries_loss = []
    for country in countries:
        country_obs = observations[observations["Country"] == country]
        countries_loss.append(np.round(poli_fit_2._loss(country_obs["DayOfYear"],
                                                        country_obs["Temp"])))

    fig5 = px.bar(x=countries, y=countries_loss)
    fig5.update_layout(xaxis_title="Country",
                       yaxis_title="Loss",
                       title="The Loss Recorded As A Function Of The Country",
                       font=dict(family="Arial", size=18))

    fig5.show()

