import pandas as pd
import vixstructure.models as models


def get_model(hidden_layers, past_days, days_to_future):
    model = models.naive_fully_connected(hidden_layers, past_days, days_to_future)
    model.compile("SGD", "mean_squared_error")
    return model


def get_data():
    # Load and merge the data.
    xm_settle = pd.read_csv("8_m_settle.csv", usecols=range(1, 10),
                            parse_dates=[0], header=0, index_col=0, na_values=0)
    vix = pd.read_csv("vix.csv", parse_dates=[0], header=0, index_col=0)
    vix = vix_index["Adj Close"]
    trainingdata = pd.merge(vix, xm_settle, left_index=True, right_index=True)
    # trainingdata has now the shape (N, 9)
    # First fill the NaN values and extract a numpy array.
    trainingdata = trainingdata.fillna(0).values
    return trainingdata  # TODO Write a generator for this.


if __name__ == "__main__":
    model = get_model(5, 7, 7)
