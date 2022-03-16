from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"

def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    uni_gaus = UnivariateGaussian()
    random_samples = np.random.normal(10, 1, 1000)
    uni_gaus.fit(random_samples)
    print("(" + str(uni_gaus.mu_) + "), (" + str(uni_gaus.var_) + ")")

    # Question 2 - Empirically showing sample mean is consistent
    samples = np.arange(10, 1000, 10)
    mean = []
    for i in range(10, 1000, 10):
        uni_gaus.fit(random_samples[0: i])
        mean.append(np.abs(10 - uni_gaus.mu_))

    fig1 = go.Figure(data=go.Bar(x=samples, y=mean, marker=dict(color='blue', line_width=1)))
    fig1.update_layout(xaxis_title="Sample Size", yaxis_title="Distance",
                      title="Expectation Distance Graph", font=dict(family="Arial", size=30))
    fig1.show()


    # Question 3 - Plotting Empirical PDF of fitted model
    random_samples_pdf = uni_gaus.pdf(random_samples)
    fig2 = go.Figure(data=go.Scatter(x=random_samples, y=random_samples_pdf, mode='markers',
                                    marker=dict(color='blue', line_width=1)))
    fig2.update_layout(xaxis_title="Sample", yaxis_title="PDF", title="PDF Graph",
                      font=dict(family="Arial", size=30))
    fig2.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    multi_gaus = MultivariateGaussian()
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    random_samples = np.random.multivariate_normal(mu, cov, 1000)
    multi_gaus.fit(random_samples)

    print(multi_gaus.mu_)
    print(multi_gaus.cov_)

    # Question 5 - Likelihood evaluation
    f1, f3 = np.linspace(-10, 10, 200), np.linspace(-10, 10, 200)

    total_length = len(f1) * len(f3)
    new_mu = np.transpose(np.array([np.repeat(f1, len(f3)), np.zeros(total_length),
                                               np.tile(f3, len(f1)), np.zeros(total_length)]))

    log_likelihood = []
    for expection in new_mu:
        log_likelihood.append(multi_gaus.log_likelihood(expection, cov, random_samples))

    fig3 = go.Figure(data=go.Heatmap(x=np.tile(f3, len(f3)), y=np.repeat(f1, len(f1)), z=log_likelihood,
                                     colorscale='ice', showlegend=False))
    fig3.update_layout(xaxis_title="f1", yaxis_title="f3", title="Log-Likelihood Heatmap Graph",
                       font=dict(family="Arial", size=30))

    fig3.show()

    # Question 6 - Maximum likelihood
    print(np.around(new_mu[np.argmax(log_likelihood)], 3))

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()