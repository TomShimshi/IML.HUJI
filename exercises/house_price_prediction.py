from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
import time

def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    data = pd.read_csv(filename).dropna()
    data = data.drop_duplicates()


    # Delete uninteresting features
    for feature in ["id", "date", "lat", "long"]:
        data = data.drop(feature, axis=1)

    # Delete invalid rows
    for feature in ["price", "bedrooms", "sqft_living", "sqft_lot", "floors", "yr_built",
                    "zipcode", "sqft_living15", "sqft_lot15"]:
        data = data[data[feature] > 0]

    for feature in ["bathrooms", "sqft_basement", "yr_renovated"]:
        data = data[data[feature] >= 0]

    data = data[data["waterfront"].isin([0,1])]
    data = data[data["view"].isin(range(5))]
    data = data[data["condition"].isin(range(1,6))]
    data = data[data["grade"].isin(range(1,14))]

    # One hot vector
    data["renovated"] = (data["yr_renovated"] / 1000).astype(int)
    data = data.drop("yr_renovated", axis=1)

    data["zipcode"] = data["zipcode"].astype(int)
    data = pd.get_dummies(data, prefix='zipcode', columns=['zipcode'])

    data.insert(loc=0, column="intercept", value=1)

    response = data["price"]
    data = data.drop("price", axis=1)

    return (data, response)


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    X = X.drop("intercept", axis=1)
    deviation_y = np.std(y)
    for feature in X.columns:
        feature_cov = np.cov(X[feature], y) / (np.std(X[feature]) * deviation_y)
        feature_pearson = feature_cov[0, 1]

        fig1 = px.scatter(pd.DataFrame({'x': X[feature], 'y': y}), x='x', y='y', trendline="ols")
        fig1.update_layout(xaxis_title="{0}".format(feature), yaxis_title="Observation",
                           title="Pearson correlation between {0} feature and the response \
                           <br> The value is: {1}".format(feature, np.round(feature_pearson, 3)),
                           font=dict(family="Arial", size=18))
        fig1.write_image(output_path % feature)


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of housing prices dataset
    observations, response = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(observations, response, "Pearson_Correlation/%s.png")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(observations, response)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    linear_regression = LinearRegression()
    num_rows = len(train_y)
    means_arr = []
    plus_vars_arr = []
    minus_vars_arr = []
    percentages = list(range(10, 101))
    samples_num = train_X.shape[0]
    for percentage in percentages:
        to_percentage = percentage / 100
        res = []
        curr_mean, curr_std = 0, 0
        for _ in range(10):
            indexes_to_take = np.random.randint(0, samples_num - 1, int((to_percentage) * samples_num))
            new_train_X = np.asarray(train_X)[indexes_to_take]
            new_train_Y = np.asarray(train_y)[indexes_to_take]
            linear_regression._fit(new_train_X, new_train_Y)
            res.append(linear_regression._loss(test_X.to_numpy(), test_y.to_numpy()))
        curr_mean = np.mean(res)
        curr_std = np.std(res)
        means_arr.append(curr_mean)
        plus_vars_arr.append(curr_mean + (2 * curr_std))
        minus_vars_arr.append(curr_mean - (2 * curr_std))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=percentages,
                             y=means_arr,
                             mode='lines+markers',
                             name='Mean'
                             ))

    fig.add_traces([go.Scatter(x=percentages, y=plus_vars_arr, fill=None, mode="lines",
               line=dict(color="lightgrey"), showlegend=False),
                    go.Scatter(x=percentages, y=minus_vars_arr, fill=None, mode="lines",
                               line=dict(color="lightgrey"), name='Confidence Interval')])
    fig.update_layout(xaxis_title="Percentage", yaxis_title="Mean Loss", title="Linear Regression Model",
                       font=dict(family="Arial", size=18))
    fig.show()





