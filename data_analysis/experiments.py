import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
import seaborn as sns
from data_analysis.dataset_analysis import create_listeners
from sklearn.linear_model import LinearRegression

"""
How does the intensity of listening to a given genre of music 
affect the reduction of mental disorders?
"""


def generate_regression():
    classical_listeners, edm_listeners, metal_listeners, pop_listeners = create_listeners()

    regression_visualization(classical_listeners, "Classical", "Anxiety")
    regression_visualization(classical_listeners, "Classical", "Depression")
    regression_visualization(classical_listeners, "Classical", "Insomnia")

    regression_visualization(edm_listeners, "EDM", "Anxiety")
    regression_visualization(edm_listeners, "EDM", "Depression")
    regression_visualization(edm_listeners, "EDM", "Insomnia")

    regression_visualization(metal_listeners, "Metal", "Anxiety")
    regression_visualization(metal_listeners, "Metal", "Depression")
    regression_visualization(metal_listeners, "Metal", "Insomnia")

    regression_visualization(pop_listeners, "Pop", "Anxiety")
    regression_visualization(pop_listeners, "Pop", "Depression")
    regression_visualization(pop_listeners, "Pop", "Insomnia")


def regression_visualization(set, title, disorder):
    sns.regplot(x=set[disorder], y=set['Hours per day'])
    plt.title(f'{title} Listeners')
    plt.xlabel(f'{disorder}')
    plt.ylabel('Hours per Day')
    plt.savefig(f"plots/regression/{title}_{disorder}_reg.png")
    plt.show()


def generate_regression_data():
    classical_listeners, edm_listeners, metal_listeners, pop_listeners = create_listeners()

    regression_data(classical_listeners['Anxiety'], classical_listeners['Hours per day'], "--CLASSICAL - ANXIETY--")
    regression_data(classical_listeners['Depression'], classical_listeners['Hours per day'], "--CLASSICAL - DEPRESSION--")
    regression_data(classical_listeners['Insomnia'], classical_listeners['Hours per day'], "--CLASSICAL - INSOMNIA--")
    print("\n")
    regression_data(edm_listeners['Anxiety'], edm_listeners['Hours per day'], "--EDM - ANXIETY--")
    regression_data(edm_listeners['Depression'], edm_listeners['Hours per day'], "--EDM - DEPRESSION--")
    regression_data(edm_listeners['Insomnia'], edm_listeners['Hours per day'], "--EDM - INSOMNIA--")
    print("\n")
    regression_data(metal_listeners['Anxiety'], metal_listeners['Hours per day'], "--METAL - ANXIETY--")
    regression_data(metal_listeners['Depression'], metal_listeners['Hours per day'], "--METAL - Depression--")
    regression_data(metal_listeners['Insomnia'], metal_listeners['Hours per day'], "--METAL - INSOMNIA--")
    print("\n")
    regression_data(pop_listeners['Anxiety'], pop_listeners['Hours per day'], "--POP - ANXIETY--")
    regression_data(pop_listeners['Depression'], pop_listeners['Hours per day'], "--POP - Depression--")
    regression_data(pop_listeners['Insomnia'], pop_listeners['Hours per day'], "--POP - INSOMNIA--")


def regression_data(x, y, title):
    # Fitting the linear regression model
    coefficients = np.polyfit(x, y, 1)
    a = coefficients[0]
    b = coefficients[1]

    # Calculating the predicted values
    y_pred = a * x + b

    mse = mean_squared_error(y, y_pred)

    print(title)
    print("Model Parameters:")
    print("a =", a)
    print("b =", b)
    print("Equation of the line: y =", a, "* x +", b)
    print("Mean Squared Error (MSE):", mse)


def generate_estimators():
    classical_listeners, edm_listeners, metal_listeners, pop_listeners = create_listeners()

    estimators(classical_listeners['Anxiety'], classical_listeners['Hours per day'], classical_listeners, "Classical", 6)
    estimators(classical_listeners['Depression'], classical_listeners['Hours per day'], classical_listeners, "Classical", 7)
    estimators(classical_listeners['Insomnia'], classical_listeners['Hours per day'], classical_listeners, "Classical", 8)

    estimators(edm_listeners['Anxiety'], edm_listeners['Hours per day'], edm_listeners, "EDM", 6)
    estimators(edm_listeners['Depression'], edm_listeners['Hours per day'], edm_listeners, "EDM", 7)
    estimators(edm_listeners['Insomnia'], edm_listeners['Hours per day'], edm_listeners, "EDM", 8)

    estimators(metal_listeners['Anxiety'], metal_listeners['Hours per day'], metal_listeners, "Metal", 6)
    estimators(metal_listeners['Depression'], metal_listeners['Hours per day'], metal_listeners, "Metal", 7)
    estimators(metal_listeners['Insomnia'], metal_listeners['Hours per day'], metal_listeners, "Metal", 8)

    estimators(pop_listeners['Anxiety'], pop_listeners['Hours per day'], pop_listeners, "Pop", 6)
    estimators(pop_listeners['Depression'], pop_listeners['Hours per day'], pop_listeners, "Pop", 7)
    estimators(pop_listeners['Insomnia'], pop_listeners['Hours per day'], pop_listeners, "Pop", 8)


def estimators(X, Y, set, title, disorder_index):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

    # Convert X_train, Y_train and X_test to numpy arrays
    X_train = np.array(X_train).reshape(-1, 1)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test).reshape(-1, 1)

    print(f"--{title} - {set.columns[disorder_index]}--")

    # ========= Linear Model =========
    model_lin = LinearRegression()
    model_lin.fit(X_train.reshape(-1, 1), Y_train)

    # ===== Generalized Linear Model =====
    model_GLM = LinearRegression()
    gen_features = PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)
    model_GLM.fit(gen_features.fit_transform(X_train.reshape(-1, 1)), Y_train)

    print(f'Generalized Linear Model parameters: {np.round(model_GLM.coef_, 4)}, {np.round(model_GLM.intercept_, 5)}')
    MSE_GLM = mean_squared_error(Y_test, model_GLM.predict(gen_features.fit_transform(X_test.reshape(-1, 1))))
    print(f'Mean Squared Error of Generalized Linear Model: {MSE_GLM:0.3}')

    # ==== Support Vector Machine ====
    # SVR for regression, SVC for classification
    model_svr = SVR(kernel='rbf', gamma='scale', C=1)

    model_svr.fit(X_train.reshape(-1, 1), Y_train)
    MSE_SVR = mean_squared_error(Y_test, model_svr.predict(X_test.reshape(-1, 1)))
    print(f'Mean Squared Error of Support Vector Regression (SVR) Model: {MSE_SVR:0.3} \n')

    estimators_visualization(X, Y, X_train, Y_train, X_test, Y_test, title, disorder_index, model_lin, model_GLM, model_svr)

def estimators_visualization(X, Y, X_train, Y_train, X_test, Y_test, title, disorder_index, model_lin, model_GLM, model_svr):
    # Predictions of all models for the entire range of X-axis
    os_x = np.linspace(start=X.min(), stop=X.max(), num=300)
    y_lin_pred = model_lin.predict(os_x.reshape(-1, 1))
    y_GLM_pred = model_GLM.predict(PolynomialFeatures(degree=2, include_bias=True, interaction_only=False).fit_transform(os_x.reshape(-1, 1)))
    y_svr_pred = model_svr.predict(os_x.reshape(-1, 1))

    # Visualization
    plt.figure(figsize=(10, 7))
    plt.scatter(X_train, Y_train, label='Training Data', alpha=0.7)
    plt.scatter(X_test, Y_test, edgecolor='black', facecolor='none', label='Testing Data')
    plt.plot(os_x, y_lin_pred, label='Linear Model', color='tab:orange')
    plt.plot(os_x, y_GLM_pred, label='Generalized Linear Model', color='tab:red')
    plt.plot(os_x, y_svr_pred, label='Support Vector Regression (SVR) Model', color='tab:green')
    plt.title(f'{title} Listeners')
    plt.xlabel(set.columns[disorder_index], fontsize=14)
    plt.ylabel(set.columns[0], fontsize=14)
    plt.legend(fontsize=12, shadow=True, loc='upper left')
    plt.ylim([Y.min() - 0.1, Y.max() + 0.5])
    plt.savefig(f"plots/estimators/{title}_{set.columns[disorder_index]}_est.png")
    plt.show()


def generate_normal_distribution():
    classical_listeners, edm_listeners, metal_listeners, pop_listeners = create_listeners()

    normal_distribution_visualization(classical_listeners['Anxiety'], "Classical", "Anxiety")
    normal_distribution_visualization(classical_listeners['Depression'], "Classical", "Depression")
    normal_distribution_visualization(classical_listeners['Insomnia'], "Classical", "Insomnia")

    normal_distribution_visualization(edm_listeners['Anxiety'], "EDM", "Anxiety")
    normal_distribution_visualization(edm_listeners['Depression'], "EDM", "Depression")
    normal_distribution_visualization(edm_listeners['Insomnia'], "EDM", "Insomnia")

    normal_distribution_visualization(metal_listeners['Anxiety'], "Metal", "Anxiety")
    normal_distribution_visualization(metal_listeners['Depression'], "Metal", "Depression")
    normal_distribution_visualization(metal_listeners['Insomnia'], "Metal", "Insomnia")

    normal_distribution_visualization(pop_listeners['Anxiety'], "Pop", "Anxiety")
    normal_distribution_visualization(pop_listeners['Depression'], "Pop", "Depression")
    normal_distribution_visualization(pop_listeners['Insomnia'], "Pop", "Insomnia")


def normal_distribution(param, x):
    mu, sigma = param
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


def normal_distribution_visualization(X, title, disorder):
    mu, sigma = X.mean(), X.std()
    os_x = np.linspace(X.min() - 10, X.max() + 10, num=300)
    os_y = normal_distribution([mu, sigma], os_x)

    plt.plot(os_x, os_y)
    plt.plot(X, np.zeros_like(X), 'o',
             markersize=10, alpha=0.5, markerfacecolor='tab:orange', markeredgecolor='tab:blue',
             label='data')
    plt.legend(fontsize=12, shadow=True)
    plt.savefig(f"plots/normal_distribution/{title}_{disorder}_nd.png")
    plt.show()

    print(f"--{title} - {disorder}--")
    print(f"Dataset size = {X.size}")
    print(f'Test Shapiro, p-value = {stats.shapiro(X)[1]}')
    print(f'Test D’Agostino and Pearsona, p-value = {stats.normaltest(X)[1]}\n')
