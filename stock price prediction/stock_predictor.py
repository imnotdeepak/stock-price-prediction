import requests
import numpy as np
import tensorflow as tf
from hmmlearn import hmm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split

API_KEY = ""
INTERVAL = "DAILY"

def get_stock_data(symbol):
    """
    Returns the stock data for a given symbol using daily intervals.
    note in the URL - &outputsize can be set to either ---
        full - Outputs data on the set "interval" for the last 20 years
        compact - Outputs data on the set "interval" for the last 100 data points
    We can also set the interval var from DAILY to WEEKLY or MONTHLY
    """
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_{INTERVAL.upper()}_ADJUSTED&symbol={symbol}&outputsize=full&apikey={API_KEY}" 
    response = requests.get(url)

    while response.status_code != 200:
        print("Error: ", response.status_code, " - response request unsuccessful")
        symbol = input("Stock Symbol entered Invalid. Please Reenter Stock Symbol\n").upper()
            
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_{INTERVAL.upper()}_ADJUSTED&symbol={symbol}&outputsize=full&apikey={API_KEY}" 
        response = requests.get(url)
    
    stock_data = response.json()

    return stock_data

def get_stock_historical_prices(data):
    """
    returns a numpy array containing the historical prices set on a specified interval
    from the extraced json respose
    """
    # change the case of the Interval value to match 
    first_letter = INTERVAL[0]
    rest_of_word = INTERVAL[1:].lower()
    Interval = first_letter + rest_of_word

    prices = list()
    for date, info in data[f"Time Series ({Interval})"].items():
        # gets the closing price of the stock for the specified interval as float
        price = float(info["4. close"])
        prices.append(price)

    # Convert the prices list to a numpy array
    prices = np.array(prices)
    
    return prices

# This is a function that calls all the other calculations and returns their output
def get_stock_direction_and_percentage_change(prices):
    """
    uses the stock prices to train the AI using two functions and
    determines whether the stock price will go up (1) or down (0)
    """

    # cuts the numpy array down to the last 5 years of historical data
    prices_last_5_years = prices[-(5*252):]
    
    # once complete, return the direction and percentage change
    return get_stock_direction_change(prices_last_5_years), get_stock_percentage_change(prices_last_5_years), get_stock_percentage_change_neural_net(prices_last_5_years)

# This is a function to learn from the given data to determine the direction of price change
def get_stock_direction_change(prices_x_years):
    """
    This will run the data through an algorithm and determine the direction of change
    given the historical pricing data fed to it
    """
    # placeholder variables
    # calculate the change in price for each day
    difference = np.diff(prices_x_years)
    # exclude the first element from the array to make data for next day
    up_or_down = np.where(difference[1:] > 0.0, 1, 0) # if the price is > 0, return 1, else 0
    
    logistic_regression = LogisticRegression(random_state=69) # create logistic regression model
    # input for current day's data by excluding the last element in the array
    x = difference[:-1].reshape(-1,1) 

    # Use LR to train
    # Split the data into training and testing data for the prediction model 
    x_train, x_test, y_train, y_test = train_test_split(x, up_or_down, test_size=.2, random_state=69, shuffle=False)
    logistic_regression.fit(x_train, y_train)
    
    # Make a prediction with LR Model for next day (modified from ChatGPT)
    y_prediction = logistic_regression.predict(x_test)
    # compare the actual data vs the prediction to determine the accuracy of the model
    # I got this idea from ChatGPT
    accuracy = np.mean(y_prediction == y_test)
    print("Calculated Accuracy for LR Model:",accuracy)
    # return whether the price went up or down (binary)
    return y_prediction[0]

def get_stock_percentage_change(prices_x_years):
    """
    This will run the data through an algorithm and determine the percentage of change
    given the historical pricing data fed to it
    """

    # Define model
    model = hmm.GaussianHMM(n_components=4, n_iter=100)

    # Fit model to data
    model.fit(prices_x_years.reshape(-1, 1))

    # Predict hidden states using model
    hidden_states = model.predict(prices_x_years.reshape(-1, 1))

    # Compute mean percentage change for each hidden state
    mean_changes = np.zeros(model.n_components)
    for i in range(model.n_components):
        mean_changes[i] = np.mean(np.diff(prices_x_years[hidden_states == i]) / prices_x_years[hidden_states == i][:-1])

    # Predict next hidden state based on current state
    current_state = hidden_states[-1]
    next_state = model.predict(prices_x_years[-1].reshape(-1, 1))[0]

    # Predict the percentage change in stock prices based on the current and next hidden states
    percentage_change = mean_changes[next_state] - mean_changes[current_state]

    return percentage_change
# calculate percentage change in stock price based on neural network

def get_stock_percentage_change_neural_net(prices_x_years):
    # need to preprocess the price data
    data_resizer = MinMaxScaler(feature_range=(0,1))
    resized_data = data_resizer.fit_transform(prices_x_years.reshape(-1,1))
    # need to split data into training set and testing set
    train_data, test_data = train_test_split(resized_data, test_size=0.2, random_state=69)

    # create a dataset to loop through and return two np arrays 
    def create_dataset(data, seq_length):
        X = []
        y = []
        for i in range(len(data) - seq_length - 1):
            X.append(data[i:(i+seq_length), :])
            y.append(data[(i+seq_length), 0])
        return np.array(X), np.array(y).reshape(-1,1)
    
    steps = 10
    # fill training and test vars with new datasets  
    train_X, train_y = create_dataset(train_data, steps)
    test_X, test_y = create_dataset(test_data, steps)
    # reshape the training and test data to be 2-dimensional
    #train_X = np.reshape(-1, (train_X.shape[0],train_X.shape[1], 1))
    #test_X = np.reshape(-1, (test_X.shape[0], test_X.shape[1], 1))
    # create the neural network model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(1000, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(260))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1))
    # compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit model to data
    model.fit(train_X, train_y, epochs=50, batch_size=64, verbose=1)

    # make a prediction of the percentage change
    predicted_percentage_change = model.predict(test_X)

    # invert scaling
    predicted_percentage_change = data_resizer.inverse_transform(predicted_percentage_change)[0][0]
    test_y = data_resizer.inverse_transform(test_y)
    
    # check model accuracy
    accuracy = np.mean((predicted_percentage_change - test_y)**2)
    print("Neural Net Calculated Mean-Squared Error", accuracy)
    accuracy = np.mean(np.abs(predicted_percentage_change - test_y))
    print("Neural Net Calculated Mean-Absolute Error", accuracy)

    return predicted_percentage_change


# This will check the stock based on the symbol the user enters in the command line, 

stock_symbol = input("Please enter a stock symbol you want to check\n").upper()
historical_prices = get_stock_historical_prices(get_stock_data(stock_symbol))
stock_direction, percentage_change, percentage_change_nn = get_stock_direction_and_percentage_change(historical_prices)
if (stock_direction == 1):
    print ("\nPredicted direction of change is increasing using Logistic Regression")
else:
    print ("\nPredicted direction of change is decreasing using Logistic Regression")
print("Using HMM: Predicted Percentage Change of",stock_symbol,"-", percentage_change, "%")
print("Using Neural Net: Predicted Percentage Change of", percentage_change_nn, "%")
