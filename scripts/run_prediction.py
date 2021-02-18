from statsmodels.regression.linear_model import OLSResults

def load_model(model_name):
    return OLSResults.load(f"models/{model_name}.pickle")

def predict(years_in_future, results):
    current_value = results.predict([1, 2457156.5])
    prediction = results.predict([1, 2457156.5+360*years_in_future])
    return prediction

if __name__ == '__main__':
    prediction = predict(2, load_model('price_over_time'))
    print(prediction)