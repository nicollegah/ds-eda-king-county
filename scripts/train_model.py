import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sms
import seaborn as sns
import matplotlib.pyplot as plt

def find_top_n_growth_zips(paramdict, n):
    sorted_dict = dict(sorted(paramdict.items(), key=lambda price: price[1], reverse = True))
    return list(sorted_dict.keys())[:n]

def fit_data(df, zipcode):
    zipdata = df[df['zipcode']==zipcode].groupby('date')['price'].mean()
    zipdata = pd.DataFrame(zipdata)
    zipdata['jdate'] = zipdata.index.to_julian_date()
    x = sms.add_constant(zipdata['jdate'])
    model = sms.OLS(zipdata['price'], x)
    results = model.fit()
    return results

def save_model(results, model_name):
    results.save(f"models/{model_name}.pickle")

def load_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df

if __name__ == "__main__":
    df = load_data('data/King_County_House_prices_dataset.csv')
    zipcode = 98188
    results = fit_data(df, zipcode)
    save_model(results, 'price_over_time')