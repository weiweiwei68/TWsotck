import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import pandas as pd
from plotly.graph_objects import Scatter, Layout
from plotly import plot




def load_data(df, dfp, sequence_length=10, split=0.8):
    #convert features from dataframe to float
    scaler = MinMaxScaler()
    data_all = np.array(df).astype(float)
    data_all = scaler.fit_transform(data_all)

    #convert labels from dataframe to float
    datap_all = np.array(dfp).astype(float)
    print("data_all's shape", datap_all.shape)
    #scale labels between 0 and 1
    datap_all = scaler.fit_transform(datap_all)

    #split every sequence_length's days into a sector, label is the day after the sector

    data = []
    datap = []
    for i in range(len(data_all) - sequence_length):
        data.append(data_all[i : i + sequence_length])
        datap.append(datap_all[i + sequence_length])

    x = np.array(data).astype("float64")
    y = np.array(datap).astype("float64")

    #Split training set and testing set
    split_boundary = int(x.shape[0] * split)
    train_x = x[:split_boundary]
    test_x = x[split_boundary:]

    train_y = y[:split_boundary]
    test_y = y[split_boundary:]

    return train_x, train_y, test_x, test_y


#Define the LSTM model
def build_model():
    model = Sequential()
    model.add(LSTM(input_shape=(10, 4), units=256, unroll=False))
    model.add(Dense(units=1))
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
    return model

def train_model(model, train_x, train_y, test_x, test_y):
    model.fit(train_x, train_y, batch_size=100, epochs=300, validation_split=0.1)
    predict = model.predict(test_x)
    predict = np.reshape(predict, (predict.size))

    return predict

#cancel diaplaying pandas imformation reset alarm
pd.options.mode.chained_assignment = None

filename = "stock_data_4909.csv"
df = pd.read_csv(filename, encoding="big5")
ddtrain = df[["CLOSE", "HIGH", "LOW", "CAPACITY"]]
ddprice = df[["CLOSE"]]

train_x, train_y, test_x, test_y = load_data(ddtrain, ddprice, sequence_length=10, split=0.8)

model = build_model()
predict_y = train_model(model, train_x, train_y, test_x, test_y)

plt.plot(test_y, color='red', label='Real stock price')
plt.plot(predict_y, color='blue', label='Predicted stock price')
plt.title('Stock Price Predection')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.savefig('Stock_price_prediction.png')
plt.show()
'''
#Plot the graphy
dd2 = pd.DataFrame({"predict":list(predict_y), "label":list(test_y)})
dd2["predict"] = np.array(dd2["predict"]).astype("float64")
dd2["label"] = np.array(dd2["label"]).astype("float64")

data = [
    Scatter(y = dd2["predict"], name = "Prediction", line = dict(color = "blue", dash = "dot")),
    Scatter(y = dd2["label"], name = "CLOSE", line = dict(color = "red"))
]
plot({"data":data, "layout":Layout(title = "2376 price prediction")}, kind='line')
'''