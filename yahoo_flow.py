import pymongo
import yfinance as yf

from pandas_datareader import data as pdr
from datetime import datetime, timedelta

import torch
import random
import numpy as np

from ez4cast import Trainer
from ez4cast.model.tempflow.tempflow_estimator import TempFlowEstimator
from gluonts.dataset.multivariate_grouper import ListDataset, MultivariateGrouper

import prefect
from prefect.run_configs import UniversalRun
from prefect.engine.results import LocalResult

from prefect import task, Flow
from prefect.storage import GitHub

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
logger = prefect.context.get("logger")
ticker_dict = {
    'AAPL': 'Apple',
    'MSFT': 'Microsoft',
    'GOOG': 'Google',
    'AMZN': 'Amazon',
    'FB': 'Facebook',
    'NVDA': 'Nvidia',
    'INTC': 'Intel',
    'BABA': 'Alibaba',
    'TSLA': 'Tesla',
    '^GSPC': 'S&P_500',
    '^IXIC': 'Nasdaq',
    '^DJI': 'Dow_Jones',
}


@task
def fetch_data_from_yahoo():

    data_yahoo = None
    get_col = 'Close'
    yf.pdr_override()
    past_14_days = datetime.now() - timedelta(days=7)

    for index_ticker, ticker in enumerate(ticker_dict):
        ticker_data = pdr.get_data_yahoo(ticker, period="max", interval = "5m", start=past_14_days.strftime('%Y-%m-%d'))
        if index_ticker == 0:
            data_yahoo = ticker_data[[get_col]].copy()
            data_yahoo.columns = [ticker]
            continue
        data_yahoo[ticker] = ticker_data[get_col].copy()

    data_yahoo.drop(data_yahoo.tail(1).index,inplace=True)
    data_yahoo.index = data_yahoo.index.tz_localize(None)
    data_yahoo = data_yahoo[:-1]
    logger.info('Fetched data from yahoo')
    return data_yahoo


@task
def train_model(input_df):
    col_len = input_df.columns.__len__()
    _freq = '5M'
    train_data_by_c = ListDataset(
        [{
            'start': input_df.index[0],
            'target': input_df[cols],
        } for cols in input_df.columns],
        freq=_freq,
        one_dim_target=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_grouper = MultivariateGrouper(col_len)
    dataset_train_by_c = train_grouper(train_data_by_c)
    model_ts = TempFlowEstimator(
        input_size=38,
        freq=_freq,
        prediction_length=1,
        context_length=2,
        flow_type='RealNVP',
        target_dim=col_len,
        trainer=Trainer(epochs=10, device=device, learning_rate=1e-3, batch_size=32),
    )

    ptd = model_ts.train_model(
        training_data=dataset_train_by_c, input_net=None
    )
    list_output = list()
    output = list(ptd.predictor.predict(dataset_train_by_c, 1000))[0].samples
    for i in output.T:
        list_output.append(
            np.mean(i[0])
        )
    logger.info(f"Predict value: {list_output}")
    try:
        mongo_client = pymongo.MongoClient("mongodb://14.241.231.87:27017/")
        my_db = mongo_client['flowdb']
        my_collection = my_db['flowdb']
        new_doc = dict()
        for idx, i in enumerate(ticker_dict):
            new_doc[ticker_dict[i]] = list_output[idx]
        response = my_collection.insert_one(new_doc)
        logger.info(response)
        mongo_client.close()
    except:
        pass


with Flow(
        "yahoo_predict",
        result=LocalResult(),
        storage=GitHub(
            repo="hoangperry/yahoo-predict-flow",
            path="yahoo_flow.py",
        ),
        run_config=UniversalRun(labels=["hoangai"]),
) as flow:
    df = fetch_data_from_yahoo()
    result = train_model(df)
    push_data_to_db(result)

flow.register(project_name="test")
