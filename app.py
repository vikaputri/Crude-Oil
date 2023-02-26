from flask import Flask, render_template
import yfinance as yf
import datetime as dt
from datetime import date
from dateutil.relativedelta import relativedelta 
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import io
import urllib, base64
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
from pmdarima.arima import auto_arima
plt.style.use('fivethirtyeight')
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

app = Flask(__name__)


@app.route('/')
def hello():
    #Crawling Data
    today = date.today()
    one_years = date.today() - relativedelta(months=+36)
    df = yf.download('BZ=F', start=one_years, end=dt.datetime.now(), progress=False)

    #Untuk menampilkan data harga selama 2 minggu
    dogecoin_price = df.tail(14)
    dogecoin_price = dogecoin_price.reset_index()
    dogecoin_price['Date'] = pd.to_datetime(dogecoin_price['Date']).dt.strftime('%m-%d-%Y')

    #Prepocessing
    del df['Open'], df['High'], df['Low'], df['Adj Close'], df['Volume']

    #Split Dataset
    train_size = int(len(df) * 0.9)
    test_size = len(df) - train_size
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]
    #df_train, df_test = train_test_split(df, test_size = 0.10, train_size =0.90)

    #Pemodelan
    model = auto_arima(df, start_p=0, d=0, start_q=0)
    get_parametes = model.get_params()
    order = get_parametes.get('order')
    mod = sm.tsa.statespace.SARIMAX(df,order=(order))
    results = mod.fit()

    #Evaluasi dengan Visualisasi
    one_months = date.today() - relativedelta(days=+len(df_test))
    pred = results.get_prediction(start=pd.to_datetime(one_months), dynamic=False)
    
    pred_ci = pred.conf_int()
    hasil = pred.predicted_mean

    actual_today = df.tail(1)
    actual_today = actual_today.stack().tolist()

    prediction_today = pd.DataFrame(hasil)
    prediction_today = prediction_today.tail(1)
    prediction_today = prediction_today.stack().tolist()
 
    ax = df['2019':].plot(label='observed')
    hasil.plot(ax=ax, label='Forecast', alpha=.7, figsize=(14, 11))
    ax.fill_between(
        pred_ci.index, 
        pred_ci.iloc[:, 0],
        pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    plt.legend()
    
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    prediksivsactual =  urllib.parse.quote(string)
    
    #Evaluasi dengan RMSE
    y_forecasted = pred.predicted_mean
    y_truth = df[one_months:]
    rmse = (sqrt(mean_squared_error(y_truth['Close'], y_forecasted)))

    #Prediksi
    tommorrow = date.today()+ relativedelta(days=1)
    pred_uc = results.get_forecast(steps=7)
    pred_ci = pred_uc.conf_int()
    pred_ci.index = pd.date_range(tommorrow, periods=7, freq="D")
    output = pred_uc.predicted_mean
    output.index = pd.date_range(tommorrow, periods=7, freq="D")

    prediction_tomorrow = pd.DataFrame(output)
    prediction_tomorrow1 = prediction_tomorrow.head(1).stack().tolist()
    prediction_tomorrow7 = prediction_tomorrow.head(7).stack().tolist()

    ax = df['2021':].plot(label='observed')
    output.plot(ax=ax, label='Forecast', figsize=(14, 11))
    ax.fill_between(
        pred_ci.index,
        pred_ci.iloc[:, 0],
        pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    plt.legend()
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    prediksi30 =  urllib.parse.quote(string)

    return render_template(
        'index.html', 
        column_names=dogecoin_price.columns.values, 
        row_data=list(dogecoin_price.values.tolist()),
        zip=zip,

        prediksivsactual=prediksivsactual,  
        prediksi30=prediksi30,
        prediction_tomorrow1=prediction_tomorrow1, 
        prediction_tomorrow7=prediction_tomorrow7, 
        prediction_today=prediction_today, 
        actual_today=actual_today, 
        rmse=rmse
    )