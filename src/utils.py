import sys
import os

import pandas as pd
import plotly.express as px
import gc
import logging
import glob
import pytz

from src.models import models
import src.send_message as sm
import src.myenv as myenv
import src.calcEMA as calc_utils

from pycaret.regression.oop import RegressionExperiment
from pycaret.classification.oop import ClassificationExperiment
from binance.client import Client
from itertools import combinations
from datetime import datetime, timedelta
import threading

from sqlalchemy import desc
from sqlalchemy.sql import functions 
from sqlalchemy.orm import Session

import numpy as np

log = logging.getLogger()

def get_ix_symbol(symbol, interval, stop_loss, times_regression_profit_and_loss):
  return f'{symbol}_{interval}_SL_{stop_loss}_PnL_{times_regression_profit_and_loss}'

def truncate_data_to_train(data):
  min_date = data['open_time'].min()
  max_date = data['open_time'].max()
  log.info(f'Data Loaded: Min date: {min_date} - Max date: {max_date}')
  validate_start_data = max_date - np.timedelta64(myenv.days_to_validate_train, 'D')
  
  log.info(f'Filtering all data: days_to_validate_train_data: {myenv.days_to_validate_train} days')
  min_index = data[data['open_time'] < validate_start_data].tail(myenv.rows_to_train).index.min()
  real_rows_to_train = data[data['open_time'] < validate_start_data].tail(myenv.rows_to_train).shape[0]
  real_rols_to_validate = data[data['open_time'] >= validate_start_data].shape[0]
  data = data[data.index >= min_index]

  min_date = data['open_time'].min()
  max_date = data['open_time'].max()
  log.info(f'All Data Filtered: train_start_date: {min_date} - validate_start_data: {validate_start_data} - max_date: {max_date}')
  log.info(f'rows_to_train: {real_rows_to_train} - rows_to_validate: {real_rols_to_validate} - All Data Shape: {data.shape[0]}')
  return data

def split_train_test_data(data):  
  log.info(f'Prepare Train Data...')
  max_date = data['open_time'].max()
  validate_start_data = max_date - np.timedelta64(myenv.days_to_validate_train, 'D')      
  max_index = data[data['open_time'] < validate_start_data].index.max()

  _train_data = data[data.index <= max_index]
  log.info(f'Setup model - train_data.shape: {_train_data.shape}')
  log.info(f'Setup model - train_data: label stats: \n{_train_data.groupby(myenv.label)[myenv.label].count()}')

  log.info(f'Prepare Test Data...')
  _test_data = data[(data.index > max_index)]
  log.info(f'Setup model - test_data.shape: {_test_data.shape}')
  log.info(f'Setup model - test_data: label stats: \n{_test_data.groupby(myenv.label)[myenv.label].count()}')

  return _train_data, _test_data


def get_start_timestamp_for_interval(interval):
  date = None
  match interval:
    case '1m':
      date = datetime.now(pytz.utc) - timedelta(days=30 * 3)  # 3 Months
    case '5m':
      date = datetime.now(pytz.utc) - timedelta(days=30 * 9)  # 9 Months
    case '15m':
      date = datetime.now(pytz.utc) - timedelta(days=365 * 2)  # 2 Years
    case '30m':
      date = datetime.now(pytz.utc) - timedelta(days=365 * 3)  # 3 Years
    case '1h':
      date = datetime.now(pytz.utc) - timedelta(days=365 * 5)  # 5 Years

  return int(date.timestamp() * 1000)


def reduce_database(interval_list=['1m', '5m', '15m', '30m', '1h']):
  for symbol in get_symbol_list():
    for interval in interval_list:
      data_file = f'{myenv.datadir}/{symbol}/{symbol}_{interval}.dat'
      parsed_date = get_start_timestamp_for_interval(interval)
      print(data_file)
      data = get_data(symbol=symbol, save_database=False, interval=interval, columns=myenv.all_klines_cols, parse_dates=False)
      data.info()
      data = data[(data['open_time'] >= parsed_date)]
      data.info()
      data.to_csv(data_file, sep=';', index=False, compression=dict(method='zip'))


def get_account_balance():
  with Session(models.engine) as session:
    ab = session.query(models.AccountBalance).order_by(desc(models.AccountBalance.id)).first()
  session.close()
  return ab.balance


def register_account_balance(profit_and_loss):
  session = Session(models.engine)
  transaction = session.begin()
  ab = session.query(models.AccountBalance).with_for_update().order_by(desc(models.AccountBalance.id)).first()
  balance = ab.balance + profit_and_loss
  new_ab = models.AccountBalance(balance=balance)
  session.add(new_ab)
  transaction.commit()
  session.close()
  return balance


def get_amount_to_invest(register=True):
  session = Session(models.engine)
  transaction = session.begin()
  ab = session.query(models.AccountBalance).with_for_update().order_by(desc(models.AccountBalance.id)).first()
  balance = ab.balance
  amount_invested = 0.0

  if balance >= myenv.default_amount_invested:
    amount_invested = myenv.default_amount_invested
  elif balance > 0 and balance < myenv.default_amount_invested:
    amount_invested = balance
  balance -= amount_invested
  if register:
    new_ab = models.AccountBalance(balance=balance)
    session.add(new_ab)
  transaction.commit()
  session.close()

  return amount_invested, balance

def get_sum_pnl():
  session = Session(models.engine)
  result = session.query(functions.sum(models.Ledger.pnl)).scalar()
  session.close()

  return result


def register_operation(params):
  session = Session(models.engine)
  transaction = session.begin()
  ledger = models.Ledger(**params)
  session.add(ledger)
  transaction.commit()
  session.close()


def get_latest_operation(symbol, interval):
  try:
    with Session(models.engine) as session:
      ledger = session.query(models.Ledger).filter(models.Ledger.symbol == symbol, models.Ledger.interval == interval).order_by(
        desc(models.Ledger.id)).first()
      return ledger
  except Exception as e:
    log.exception(e)
    return None


def get_params_operation(operation_date, symbol, interval, operation, target_margin, amount_invested, take_profit, stop_loss, purchase_price, rsi, sell_price,
                         profit_and_loss, margin_operation, strategy, balance):
  params_operation = {'operation_date': datetime.fromtimestamp(int(operation_date.astype(np.int64)) / 1000000000),
                      'symbol': symbol,
                      'interval': interval,
                      'operation': operation,
                      'target_margin': target_margin,
                      'amount_invested': amount_invested,
                      'take_profit': take_profit,
                      'stop_loss': stop_loss,
                      'purchase_price': purchase_price,
                      'sell_price': sell_price,
                      'pnl': profit_and_loss,
                      'rsi': rsi,
                      'margin_operation': margin_operation,
                      'strategy': strategy,
                      'balance': balance}
  return params_operation


def get_telegram_key():
  with open(f'{sys.path[0]}/telegram.key', 'r') as file:
    first_line = file.readline()
    if not first_line:
      raise Exception('telegram.key is empty')
  return first_line


def prepare_best_params():
  file_list = glob.glob(os.path.join(f'{myenv.datadir}/', 'resultado_simulacao_*.csv'))
  df_top_params = pd.DataFrame()
  for file_path in file_list:
    if os.path.isfile(file_path):
      df = pd.read_csv(file_path, sep=';')
    df['count_numeric_features'] = df['numeric_features'].apply(lambda x: len(x.split(',')))

    df.sort_values(['CAI', 'count_numeric_features'], ascending=[True, False], inplace=True)
    df_aux = df.tail(1)
    df_aux.insert(1, 'strategy', 'CAI')
    if df_aux['max_score'].values[0] > 0.5:
      df_top_params = pd.concat([df_top_params, df_aux], ignore_index=True)

    df.sort_values(['SOBE', 'count_numeric_features'], ascending=[True, False], inplace=True)
    df_aux = df.tail(1)
    df_aux.insert(1, 'strategy', 'SOBE')
    if df_aux['max_score'].values[0] > 0.5:
      df_top_params = pd.concat([df_top_params, df_aux], ignore_index=True)

  df_top_params.sort_values(['symbol', 'interval', 'strategy', 'estimator', 'imbalance_method'], inplace=True)
  top_paramers_filename = f'{myenv.datadir}/top_params.csv'
  log.info(f'Top Parameters save to: {top_paramers_filename}')
  df_top_params.to_csv(top_paramers_filename, sep=';', index=False)
  top_params = df_top_params.to_dict(orient='records')
  log.info(f'Top Params: \n{top_params}')
  return top_params


def get_best_parameters():
  top_parameters_filename = f'{myenv.datadir}/top_params.csv'
  if not os.path.isfile(top_parameters_filename):
    raise Exception(f'Top Parameters not found: {top_parameters_filename}')

  df_top_params = pd.read_csv(top_parameters_filename, sep=';')
  top_params = df_top_params.to_dict(orient='records')
  log.info(f'Top Parameters Loaded. Items: {len(top_params)}')
  return top_params


def get_symbol_list():
  result = []
  df = pd.read_csv(myenv.datadir + '/symbol_list.csv')
  for symbol in df['symbol']:
    result.append(symbol)
  return result


def prepare_numeric_features_list(list_of_numeric_features):
  combination_numeric_features = []
  if len(list_of_numeric_features) > 0:
    for size in range(1, len(list_of_numeric_features) + 1):
      comb = map(list, combinations(list_of_numeric_features, size))
      for c in comb:
        res = ''
        for j in c:
          res += f'{j},'
        combination_numeric_features.append(res[0:len(res) - 1])

  return combination_numeric_features


def combine_list(list_of_elements):
  combinations_list = []
  for i in range(1, len(list_of_elements) + 1):
    a = combinations(list_of_elements, i)
    for s in a:
      res = ''
      for j in s:
        res += f'{j},'
      combinations_list.append(res[0:len(res) - 1])

  return combinations_list


def increment_time(interval='1h'):
  match(interval):
    case '1min':
      return pd.Timedelta(minutes=1)
    case '3min':
      return pd.Timedelta(minutes=3)
    case '5min':
      return pd.Timedelta(minutes=5)
    case '15min':
      return pd.Timedelta(minutes=15)
    case '30min':
      return pd.Timedelta(minutes=30)
    case '1h':
      return pd.Timedelta(hours=1)
    case '2h':
      return pd.Timedelta(hours=2)
    case '4h':
      return pd.Timedelta(hours=4)
    case '6h':
      return pd.Timedelta(hours=6)
    case '8h':
      return pd.Timedelta(hours=8)
    case '12h':
      return pd.Timedelta(hours=12)
    case '1d':
      return pd.Timedelta(days=1)
    case '3d':
      return pd.Timedelta(days=3)
    case '1w':
      return pd.Timedelta(weeks=1)
    case '1M':
      return pd.Timedelta(days=30)


def get_latest_interval_day(ix_time, interval):
  _ix_aux = ix_time.replace(hour=0, minute=0, second=0, microsecond=0)
  day = ix_time.day - interval
  for i in range(day, -1, -1):
    if (i <= day) and (i % interval == 0):
      _ix_aux = _ix_aux.replace(hour=i, minute=0, second=0, microsecond=0)
      break
  return _ix_aux


def get_latest_interval_hours(ix_time, interval):
  _ix_aux = ix_time.replace(minute=0, second=0, microsecond=0)
  hour = ix_time.hour - interval
  for i in range(hour, -1, -1):
    if (i <= hour) and (i % interval == 0):
      _ix_aux = _ix_aux.replace(hour=i, minute=0, second=0, microsecond=0)
      break
  return _ix_aux


def get_latest_interval_minutes(ix_time, interval):
  _ix_aux = ix_time.replace(second=00, microsecond=00)
  min = ix_time.minute - interval
  for i in range(min, -1, -1):
    if (i <= min) and (i % interval == 0):
      _ix_aux = _ix_aux.replace(minute=i, second=00, microsecond=00)
      break
  return _ix_aux


def get_latest_close_time(interval='1h'):
  client = Client()
  time = client.get_server_time()
  ix = pd.to_datetime(time['serverTime'], unit='ms')

  match(interval):
    case '1min':
      return get_latest_interval_minutes(ix, 1)
    case '3min':
      return get_latest_interval_minutes(ix, 3)
    case '5min':
      return get_latest_interval_minutes(ix, 5)
    case '15min':
      return get_latest_interval_minutes(ix, 15)
    case '30min':
      return get_latest_interval_minutes(ix, 30)
    case '1h':
      return get_latest_interval_hours(ix, 1)
    case '2h':
      return get_latest_interval_hours(ix, 2)
    case '4h':
      return get_latest_interval_hours(ix, 4)
    case '6h':
      return get_latest_interval_hours(ix, 6)
    case '8h':
      return get_latest_interval_hours(ix, 8)
    case '12h':
      return get_latest_interval_hours(ix, 12)
    case '1d':
      return get_latest_interval_day(ix, 1)
    case '3d':
      return get_latest_interval_day(ix, 3)
    case '1w':
      return get_latest_interval_day(ix, 7)
    case '1M':
      return get_latest_interval_day(ix, 30)
  raise Exception(f'Wrong interval: {interval}.')


def date_parser(x):
  return pd.to_datetime(x, unit='ms')


def read_data(dir, sep=';', all_cols=None, use_cols=myenv.use_cols) -> pd.DataFrame:
  filenames = []

  for file in os.listdir(dir):
    if file.endswith(".csv"):
      filenames.append(os.path.join(dir, file))

  parse_dates = ['open_time']
  dataframes = []

  for filename in filenames:
    log.info(f'read_data: Start reading file: {filename}')
    df = pd.read_csv(filename, names=all_cols, parse_dates=parse_dates,
                     date_parser=date_parser, sep=sep, decimal='.', usecols=use_cols)
    dataframes.append(df)

  # Concatenate all DataFrames into a single DataFrame
  combined_df = pd.concat(dataframes, ignore_index=True)
  combined_df.sort_values(['open_time'], inplace=True)
  combined_df.reset_index(inplace=True, drop=True)
  return combined_df


def rotate_label(df, rows_to_rotate=-1, label='label_shifted', dropna=False):
  new_label = label + '_' + str(rows_to_rotate)
  df[new_label] = df[label].shift(rows_to_rotate)
  if dropna:
    df.dropna(inplace=True)

  return new_label, df


def setup_regression_model(
        data: pd.DataFrame,
        label: str,
        train_size=0.7,
        numeric_features=['open', 'high', 'low', 'volume', 'close', 'rsi'],
        date_features=['open_time'],
        use_gpu=False,
        estimator='lr',
        apply_best_analisys=False,
        fold=3,
        sort='MAE',
        verbose=False) -> [RegressionExperiment, any]:

  re = RegressionExperiment()

  setup = re.setup(data,
                   train_size=train_size,
                   target=label,
                   numeric_features=numeric_features,
                   date_features=date_features,
                   create_date_columns=["hour", "day", "month"],
                   fold_strategy='timeseries',
                   fold=fold,
                   session_id=123,
                   normalize=True,
                   use_gpu=use_gpu,
                   verbose=verbose,
                   )
  best = estimator
  if apply_best_analisys:
    log.info('setup_model: Applying best analisys...') if verbose else None
    best = setup.compare_models(sort=sort, verbose=True, exclude=['lightgbm'])

  log.info(f'setup_model: Creating model Best: [{best}]') if verbose else None
  model = setup.create_model(best, verbose=False)
  model_name_file = str(model)[0:str(model).find('(')] + '_' + label
  log.info(f'setup_model: Saving model {model_name_file}') if verbose else None
  setup.save_model(model, model_name_file)

  return setup, model


def predict(setup: RegressionExperiment,
            model: any,
            predict_data: pd.DataFrame = None,
            numeric_features=['open', 'high', 'low', 'volume', 'close', 'rsi'],
            date_features=['open_time'],
            verbose=False) -> RegressionExperiment:

  log.info(f'predict: predict.setup: \n {setup}') if verbose else None
  log.info(f'predict: predict.model: \n {model}') if verbose else None
  log.info(f'predict: predict.predict_data: \n {predict_data}') if verbose else None
  log.info(f'predict: predict.numeric_features: \n {numeric_features}') if verbose else None
  log.info(f'predict: predict.date_features: \n {date_features}') if verbose else None

  predict = None
  if predict_data is None:
    predict = setup.predict_model(model, verbose=verbose)
  else:
    predict = setup.predict_model(model, data=predict_data[date_features + numeric_features], verbose=verbose)

  return predict


def forecast(data: pd.DataFrame,
             fh: int = 1,
             train_size=0.7,
             interval='1h',
             numeric_features=['open', 'high', 'low', 'volume', 'close', 'rsi'],
             date_features=['open_time'],
             estimator='lr',
             apply_best_analisys=False,
             use_gpu=False,
             fold=3,
             ):
  list_models = {}

  _data = data.copy()
  test_data = data.tail(1).copy().reset_index(drop=True)
  log.info(f'forecast: numeric_features: {numeric_features}')

  open_time = test_data['open_time']
  df_result = pd.DataFrame()
  for i in range(1, fh + 1):
    df_predict = pd.DataFrame()
    open_time = open_time + increment_time(interval)
    df_predict['open_time'] = open_time
    log.info(f'forecast: Applying predict No: {i} for open_time: {df_predict["open_time"].values}')

    for label in numeric_features:
      if label not in list_models:
        log.info(f'forecast: Training model for label: {label}')
        target, train_data = rotate_label(_data, -1, label, True)
        setup, model = setup_regression_model(train_data, target, train_size=train_size, fold=fold,
                                              estimator=estimator, use_gpu=use_gpu, apply_best_analisys=apply_best_analisys)
        train_data.drop(columns=target, inplace=True)
        list_models[label] = {'setup': setup, 'model': model}
        log.info('forecast: Training model Done!')

      _setup = list_models[label]['setup']
      _model = list_models[label]['model']

      df = predict(_setup,
                   _model,
                   test_data if i == 1 else df_result.tail(1).copy(),
                   numeric_features,
                   date_features)

      log.info(f'forecast: Label: {label} - Predict Label: {df["prediction_label"].values[0]}')
      df_predict[label] = df['prediction_label']
      gc.collect()

    df_result = pd.concat([df_result, df_predict], axis=0)
    gc.collect()

  return df_result.sort_values('open_time').reset_index(drop=True)


def shift_test_data(predict_data: pd.DataFrame, label='close', columns=[], verbose=False):
  log.info(f'forecast: Shifting: \n {predict_data.tail(1)[columns]}') if verbose else None
  _test_data = predict_data[columns].tail(1).copy().shift(1, axis='columns')
  _test_data.drop(columns=label, inplace=True)
  _test_data['open_time'] = predict_data['open_time']
  log.info(f'forecast: Shifted: \n {_test_data.tail(1)}') if verbose else None
  return _test_data


def forecast2(data: pd.DataFrame,
              label: str = 'close',
              fh: int = 1,
              train_size=0.7,
              interval='1h',
              numeric_features=['open', 'high', 'low', 'volume', 'close', 'rsi'],
              date_features=['open_time'],
              estimator='lr',
              apply_best_analisys=False,
              use_gpu=False,
              fold=3,
              regression_times=1,
              sort='MAE',
              verbose=False,
              ):

  _data = data.copy()
  for i in range(1, regression_times + 1):
    _label, _data = rotate_label(_data, i, label)
    numeric_features.append(_label)
  _data.dropna(inplace=True)

  log.info(f'forecast: numeric_features: {numeric_features}') if verbose else None

  open_time = data.tail(1)['open_time']
  df_result = pd.DataFrame()
  setup = None
  model = None
  for i in range(1, fh + 1):
    if model is None:
      log.info(f'forecast: Training model for label: {label}') if verbose else None
      setup, model = setup_regression_model(_data, label, train_size, numeric_features, date_features,
                                            use_gpu, estimator, apply_best_analisys, fold, sort, verbose)
      log.info('forecast: Training model Done!') if verbose else None

    open_time = open_time + increment_time(interval)
    log.info(f'forecast: Applying predict No: {i} for open_time: {open_time}') if verbose else None
    predict_data = shift_test_data(_data.tail(1).copy() if i == 1 else df_result.tail(1).copy(), label=label, columns=[label] + numeric_features)
    predict_data['open_time'] = open_time

    df_predict = predict(setup, model, predict_data, numeric_features, date_features, verbose)
    df_predict['close'] = df_predict['prediction_label']

    gc.collect()

    df_result = pd.concat([df_result, df_predict], axis=0)
    gc.collect()

  return df_result.sort_values('open_time').reset_index(drop=True), model, setup


def calc_diff(predict_data, validation_data, estimator):
  start_date = predict_data["open_time"].min()  # strftime("%Y-%m-%d")
  end_date = predict_data["open_time"].max()  # .strftime("%Y-%m-%d")
  # now = datetime.now().strftime("%Y-%m-%d")

  predict_data.index = predict_data['open_time']
  validation_data.index = validation_data['open_time']

  filtered_data = validation_data.loc[(validation_data['open_time'] >= start_date) & (validation_data['open_time'] <= end_date)].copy()
  filtered_data['prediction_label'] = predict_data['prediction_label']
  filtered_data['diff'] = ((filtered_data['close'] - filtered_data['prediction_label']) / filtered_data['close']) * 100
  filtered_data.drop(columns=['open_time'], inplace=True)
  return filtered_data


def plot_predic_model(predict_data, validation_data, estimator):
  start_date = predict_data["open_time"].min()  # strftime("%Y-%m-%d")
  end_date = predict_data["open_time"].max()  # .strftime("%Y-%m-%d")

  filtered_data = calc_diff(predict_data, validation_data, estimator)

  fig1 = px.line(filtered_data, x=filtered_data.index, y=['close', 'prediction_label'], template='plotly_dark', range_x=[start_date, end_date])
  fig2 = px.line(filtered_data, x=filtered_data.index, y=['diff'], template='plotly_dark', range_x=[start_date, end_date])
  fig1.show()
  fig2.show()
  return filtered_data


def get_model_name_to_load(
        strategy,
        symbol,
        interval='1h',
        estimator=myenv.estimator,
        stop_loss=myenv.stop_loss,
        regression_times=myenv.regression_times,
        times_regression_profit_and_loss=myenv.times_regression_profit_and_loss):
  '''
  return: Last model file stored in MODELS_DIR or None if not exists. Max 999 models per symbol
  '''
  model_name = None
  for i in range(9999, 0, -1):
    aux_model_name = f'{strategy}_{symbol}_{interval}_{estimator}_SL_{stop_loss}_RT_{regression_times}_RPL_{times_regression_profit_and_loss}_{i}'
    if os.path.exists(f'{aux_model_name}.pkl'):
      model_name = aux_model_name
      break
  return model_name


def get_model_name_to_save(
        strategy,
        symbol,
        interval,
        estimator='xgboost',
        stop_loss=myenv.stop_loss,
        regression_times=myenv.regression_times,
        times_regression_profit_and_loss=myenv.times_regression_profit_and_loss):

  model_name = None
  for i in range(1, 9999):
    model_name = f'{strategy}_{symbol}_{interval}_{estimator}_SL_{stop_loss}_RT_{regression_times}_RPL_{times_regression_profit_and_loss}_{i}'
    if os.path.exists(f'{model_name}.pkl'):
      continue
    else:
      return model_name
  return model_name


def save_model(
        strategy,
        symbol,
        interval,
        model,
        experiment,
        estimator='xgboost',
        stop_loss=myenv.stop_loss,
        regression_times=myenv.regression_times,
        times_regression_profit_and_loss=myenv.times_regression_profit_and_loss):

  model_name = get_model_name_to_save(strategy, symbol, interval, estimator, stop_loss, regression_times, times_regression_profit_and_loss)
  log.info(f'save_model: Model file name: {model_name}')
  experiment.save_model(model, model_name)
  return model_name


def load_model(strategy, symbol, interval, estimator=myenv.estimator, stop_loss=myenv.stop_loss, regression_times=myenv.regression_times, times_regression_profit_and_loss=myenv.times_regression_profit_and_loss):
  model_name = get_model_name_to_load(strategy, symbol, interval, estimator, stop_loss, regression_times, times_regression_profit_and_loss)
  if model_name is None:
    return None, None
  log.info(f'load_model: Loading model: {model_name}')
  ca = ClassificationExperiment()
  model = ca.load_model(model_name, verbose=False)
  log.info(f'load_model: Model obj: {model}')

  return ca, model


def regresstion_times(df_database, regression_features=['close'], regression_times=24 * 30, last_one=False):
  log.info(f'regresstion_times: regression_features: {regression_features}')
  count = df_database.shape[0]
  features_added = []
  if last_one:
    col_ant = ''
    col_atual = ''
    for nf in regression_features:
      for i in range(1, regression_times + 1):
        if i == 1:
          col_ant = nf
          col_atual = nf + "_" + str(i)
        elif i == regression_times:
          continue
        else:
          col_ant = nf + "_" + str(i)
          col_atual = nf + "_" + str(i + 1)
        df_database.iloc[count:count + 1][col_atual] = df_database.iloc[count - 1:count][col_ant]
  else:
    # features_added.append(regression_features)
    for nf in regression_features:
      for i in range(1, regression_times + 1):
        col = nf + "_" + str(i)
        df_database[col] = df_database[nf].shift(i)
        features_added.append(col)

    df_database.dropna(inplace=True)
  return df_database, features_added


def get_max_date(df_database, start_date='2010-01-01'):
  max_date = datetime.strptime(start_date, '%Y-%m-%d')
  if df_database is not None and df_database.shape[0] > 0:
    max_date = pd.to_datetime(df_database['open_time'].max(), unit='ms')
  return max_date


def get_database(symbol, interval='1h', tail=-1, columns=['open_time', 'close'], parse_dates=True):
  database_name = get_database_name(symbol, interval)
  log.info(f'get_database: name: {database_name}')

  df_database = pd.DataFrame()
  log.info(f'get_database: columns: {columns}')
  if os.path.exists(database_name):
    if parse_dates:
      df_database = pd.read_csv(database_name, sep=';', parse_dates=myenv.date_features, date_parser=date_parser, decimal='.', usecols=columns, compression=dict(method='zip'))
    else:
      df_database = pd.read_csv(database_name, sep=';', decimal='.', usecols=columns, compression=dict(method='zip'))
    df_database = parse_type_fields(df_database, parse_dates)
    df_database = adjust_index(df_database)
    df_database = df_database[columns]
  if tail > 0:
    df_database = df_database.tail(tail).copy()
  log.info(f'get_database: count_rows: {df_database.shape[0]} - symbol: {symbol}_{interval} - tail: {tail}')
  log.info(f'get_database: duplicated: {df_database.index.duplicated().sum()}')
  return df_database


def get_database_name(symbol, interval):
  return f'{myenv.datadir}/{symbol}/{symbol}_{interval}.dat'


def download_data(save_database=True, parse_dates=False, interval='1h', start_date='2010-01-01'):
  symbols = pd.read_csv(myenv.datadir + '/symbol_list.csv')
  for symbol in symbols['symbol']:
    get_data(symbol=symbol, save_database=save_database, interval=interval, columns=myenv.all_klines_cols, parse_dates=parse_dates, start_date=start_date)


def adjust_index(df):
  df.drop_duplicates(keep='last', subset=['open_time'], inplace=True)
  df.index = df['open_time']
  df.index.name = 'ix_open_time'
  df.sort_index(inplace=True)
  return df

def remove_cols_for_klines(columns):
  cols_to_remove = ['symbol', 'rsi']
  for col in columns:
    if col.startswith('ema'):
      cols_to_remove.append(col)
  for col in cols_to_remove:
    if col in columns:
      columns.remove(col)
  return columns
  

def get_klines(symbol, interval='1h', max_date='2010-01-01', limit=1000, columns=['open_time', 'close'], parse_dates=True):
  # return pd.DataFrame()
  start_time = datetime.now()
  client = Client()
  klines = client.get_historical_klines(symbol=symbol, interval=interval, start_str=max_date, limit=limit)

  columns = remove_cols_for_klines(columns)

  # log.info('get_klines: columns: ', columns)
  df_klines = pd.DataFrame(data=klines, columns=myenv.all_klines_cols)[columns]
  df_klines = parse_type_fields(df_klines, parse_dates=parse_dates)
  df_klines = adjust_index(df_klines)
  delta = datetime.now() - start_time
  # Print the delta time in days, hours, minutes, and seconds
  log.debug(f'get_klines: shape: {df_klines.shape} - Delta time: {delta.seconds % 60} seconds')
  return df_klines


def parse_type_fields(df, parse_dates=False):
  try:
    if 'symbol' in df.columns:
      df['symbol'] = pd.Categorical(df['symbol'])

    for col in myenv.float_kline_cols:
      if col in df.columns:
        if df[col].isna().sum() == 0:
          df[col] = df[col].astype('float32')
    
    for col in myenv.float_kline_cols:
      if col in df.columns:
        if df[col].isna().sum() == 0:
          df[col] = df[col].astype('float32')


    for col in myenv.integer_kline_cols:
      if col in df.columns:
        if df[col].isna().sum() == 0:
          df[col] = df[col].astype('int16')

    if parse_dates:
      for col in myenv.date_features:
        if col in df.columns:
          if df[col].isna().sum() == 0:
            df[col] = pd.to_datetime(df[col], unit='ms')

  except Exception as e:
    log.exception(e)

  return df


def get_data(symbol, save_database=False, interval='1h', tail=-1, columns=['open_time', 'close'], parse_dates=True, updata_data_from_web=True, start_date='2010-01-01'):
  database_name = get_database_name(symbol, interval)
  log.info(f'get_data: Loading database: {database_name}')
  df_database = get_database(symbol=symbol, interval=interval, tail=tail, columns=columns, parse_dates=parse_dates)
  log.info(f'Shape database on disk: {df_database.shape}')

  log.info(f'Filtering start date: {start_date}')
  if parse_dates:
    df_database = df_database[df_database['open_time'] >= start_date]
    log.info(f'New shape after filtering start date. Shape: {df_database.shape}')

  max_date = get_max_date(df_database, start_date=start_date)
  max_date_aux = ''
  new_data = False
  if updata_data_from_web:
    log.info(f'get_data: Downloading data for symbol: {symbol} - max_date: {max_date}')
    while (max_date != max_date_aux):
      new_data = True
      log.info(f'get_data: max_date: {max_date} - max_date_aux: {max_date_aux}')
      max_date_aux = get_max_date(df_database, start_date=start_date)
      log.info(f'get_data: Max date database: {max_date_aux}')

      df_klines = get_klines(symbol, interval=interval, max_date=max_date_aux.strftime('%Y-%m-%d'), columns=columns, parse_dates=parse_dates)
      df_database = pd.concat([df_database, df_klines])
      df_database.drop_duplicates(keep='last', subset=['open_time'], inplace=True)
      df_database.sort_index(inplace=True)
      df_database['symbol'] = symbol
      max_date = get_max_date(df_database)

  if save_database and new_data:
    sulfix_name = f'{symbol}_{interval}.dat'
    if not os.path.exists(database_name.removesuffix(sulfix_name)):
      os.makedirs(database_name.removesuffix(sulfix_name))
    df_database.to_csv(database_name, sep=';', index=False, compression=dict(method='zip'))
    log.info(f'get_data: Database updated at {database_name}')

  log.info(f'New shape after get_data: {df_database.shape}')
  return df_database


def send_message(df_predict):
  message = f'Ticker: {df_predict["symbol"].values[0]} - Operação: {df_predict["prediction_label"].values[0]} - Valor Atual: {df_predict["close"].values[0]}'
  sm.send_to_telegram(message)
  log.info(f'send_message: {message}')


def set_status_PL(row, stop_loss, max_regression_profit_and_loss, prefix_col_diff, strategy='SOBE_CAI'):
  for s in range(1, max_regression_profit_and_loss + 1):
    if (strategy == 'SOBE') or (strategy == 'SOBE_CAI'):
      if row[f'{prefix_col_diff}{s}'] >= stop_loss:
        return f'SOBE_{stop_loss}'
    if (strategy == 'CAI') or (strategy == 'SOBE_CAI'):
      if row[f'{prefix_col_diff}{s}'] <= -stop_loss:
        return f'CAI_{stop_loss}'
  return 'ESTAVEL'


def regression_PnL(data: pd.DataFrame, label: str, diff_percent: float, max_regression_profit_and_loss=6, drop_na=True, drop_calc_cols=True, strategy=None):
  col = 'c_'
  diff_col = 'd_'
  cols = []
  diff_cols = []
  _data = pd.DataFrame()
  for i in range(1, max_regression_profit_and_loss + 1):
    _data[col + str(i)] = data['close'].shift(-i)
    _data[diff_col + str(i)] = 100 * ((_data[col + str(i)] - data['close']) / data['close'])
    cols.append(col + str(i))
    diff_cols.append(diff_col + str(i))

  if strategy == 'SOBE_CAI':
    _data[label + '_sobe'] = _data.apply(set_status_PL, axis=1, args=[diff_percent, max_regression_profit_and_loss, diff_col, 'SOBE'])
    _data[label + '_cai'] = _data.apply(set_status_PL, axis=1, args=[diff_percent, max_regression_profit_and_loss, diff_col, 'CAI'])
    _data[label + '_sobe'] = pd.Categorical(_data[label + '_sobe'])
    _data[label + '_cai'] = pd.Categorical(_data[label + '_cai'])
  else:
    _data[label] = _data.apply(set_status_PL, axis=1, args=[diff_percent, max_regression_profit_and_loss, diff_col, 'SOBE_CAI'])
    _data[label] = pd.Categorical(_data[label])

  if drop_calc_cols:
    _data.drop(columns=cols + diff_cols, inplace=True)

  data = pd.concat([data, _data], axis=1)
  if drop_na:
    data.dropna(inplace=True)

  return data


def regress_until_diff(data: pd.DataFrame, diff_percent: float, max_regression_profit_and_loss=6, label: str = None):
  data['close_shift_x'] = 0.0
  data['diff_shift_x'] = 0.0
  data['shift_x'] = 0
  data[label] = 'ESTAVEL'
  for row_nu in range(1, data.shape[0]):
    diff = 0
    i = 1

    while (abs(diff) <= diff_percent):
      if (i > max_regression_profit_and_loss) or ((row_nu + i) >= data.shape[0]):
        break

      close = data.iloc[row_nu:row_nu + 1]['close'].values[0]
      close_px = data.iloc[row_nu + i:row_nu + i + 1]['close'].values[0]
      diff = -100 * (close - close_px) / close
      # log.info(f'ROW_NU: {row_nu} - regresssion_times: {i} - diff: {diff}')
      i += 1

    data['close_shift_x'].iloc[row_nu:row_nu + 1] = close_px
    data['diff_shift_x'].iloc[row_nu:row_nu + 1] = diff
    data['shift_x'].iloc[row_nu:row_nu + 1] = i - 1 if i == max_regression_profit_and_loss + 1 else i

    if diff >= diff_percent:
      data[label].iloc[row_nu:row_nu + 1] = 'SOBE_' + str(diff_percent)

    elif diff <= -diff_percent:
      data[label].iloc[row_nu:row_nu + 1] = 'CAI_' + str(diff_percent)

    # end for

  data.drop(columns=['close_shift_x', 'diff_shift_x', 'shift_x'], inplace=True)
  data[label] = pd.Categorical(data[label])

  return data


def simule_trading_crypto2(df_predicted: pd.DataFrame, start_date, end_date, value: float):
  _data = df_predicted.copy()
  _data.index = _data['open_time']
  _data = _data[(_data.index >= start_date) & (_data.index <= end_date)]

  log.debug(f'Start Simule Trading: {start_date} - {end_date} - Shape: {_data.shape}')

  purchased = False
  purchase_price = 0.0
  actual_price = 0.0
  amount_invested = value
  balance = value
  take_profit_price = 0.0
  stop_loss_price = 0.0
  profit_and_loss = 0.0
  operation = ''
  margin = 0.0
  margin_operation = 0.0
  rsi = 0.0

  for row_nu in range(1, _data.shape[0]):
    actual_price = _data.iloc[row_nu:row_nu + 1]['close'].values[0]
    operation = _data.iloc[row_nu:row_nu + 1]['prediction_label'].values[0]
    amount_invested = balance
    # log.debug(f'Actual Price: $ {actual_price:.6f} - Operation: {operation} - Purchased: {purchased}')

    if 'rsi' in _data.columns:
      rsi = _data.iloc[row_nu:row_nu + 1]['rsi'].values[0]
    # Start Buy Operation
    if (not purchased):
      prediction_label = _data.iloc[row_nu:row_nu + 1]['prediction_label'].values[0]
      if prediction_label.startswith('SOBE') or prediction_label.startswith('CAI'):
        operation = prediction_label.split('_')[0]
        margin_operation = float(prediction_label.split('_')[1])

      if operation.startswith('SOBE') or operation.startswith('CAI'):
        purchased = True
        purchase_price = actual_price

        if operation.startswith('CAI'):  # Short
          take_profit_price = actual_price * (1 - margin_operation / 100)
          stop_loss_price = actual_price * (1 + (margin_operation * myenv.stop_loss_multiplier) / 100)
        elif operation.startswith('SOBE'):  # Long
          take_profit_price = actual_price * (1 + margin_operation / 100)
          stop_loss_price = actual_price * (1 - (margin_operation * myenv.stop_loss_multiplier) / 100)

        log.debug(f'\nOperation: {operation} - Perform >>> BUY <<<' +
                  f' - Actual Price: $ {actual_price:.6f} - Purchased Price: $ {purchase_price:.6f} - Amount Invested: $ {amount_invested:.2f}' +
                  f' - Take Profit: $ {take_profit_price:.6f} - Stop Loss: $ {stop_loss_price:.6f} - PnL: $ {profit_and_loss:.2f}' +
                  f' - Margin Operation: {margin_operation:.2f}% - RSI: {rsi:.2f} - Balance: $ {balance:.2f}')
        continue
    # Ends Buy Operation

    # Starts Sell Operation
    if purchased:
      perform_sell = False
      if operation.startswith('SOBE'):
        margin = (actual_price - purchase_price) / purchase_price
        if ((actual_price >= take_profit_price) or (actual_price <= stop_loss_price)):  # Long ==> Sell - Take Profit / Stop Loss
          perform_sell = True
      elif operation.startswith('CAI'):
        margin = (purchase_price - actual_price) / purchase_price
        if ((actual_price <= take_profit_price) or (actual_price >= stop_loss_price)):  # Short ==> Sell - Take Profit / Stop Loss
          perform_sell = True

      profit_and_loss = amount_invested * margin

      if perform_sell:  # Register Sell
        balance += profit_and_loss
        log.debug(f'\nOperation: {operation} - Perform >>> SELL <<<' +
                  f' - Actual Price: $ {actual_price:.6f} - Purchased Price: $ {purchase_price:.6f} - Amount Invested: $ {amount_invested:.2f}' +
                  f' - Take Profit: $ {take_profit_price:.6f} - Stop Loss: $ {stop_loss_price:.6f} - Margin: {100*margin:.2f} - PnL: $ {profit_and_loss:.2f}' +
                  f' - Margin Operation: {margin_operation:.2f}%  RSI: {rsi:.2f} - Balance: $ {balance:.2f}')
        # Reset variables
        purchased, purchase_price, amount_invested, take_profit_price, stop_loss_price, profit_and_loss, margin, margin_operation = (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Ends Sell Operation
  log.info(f'>>> Balance: $ {balance:.2f}')
  return balance


def simule_trading_crypto(df_predicted: pd.DataFrame, start_date, end_date, value: float, stop_loss=3.0, revert=False):
  _data = df_predicted.copy()
  _data.index = _data['open_time']
  _data = _data[(_data.index >= start_date) & (_data.index <= end_date)]
  saldo = value
  operacao = ''
  comprado = False
  valor_compra = 0
  valor_venda = 0
  diff = 0.0

  operacao_compra = ''
  for row_nu in range(1, _data.shape[0]):
    open_time = pd.to_datetime(_data.iloc[row_nu:row_nu + 1]['open_time'].values[0]).strftime("%Y-%m-%d %Hh")
    operacao = _data.iloc[row_nu:row_nu + 1]['prediction_label'].values[0]

    if (operacao.startswith('SOBE') or operacao.startswith('CAI')) and not comprado:
      operacao_compra = operacao
      valor_compra = _data.iloc[row_nu:row_nu + 1]['close'].values[0]
      log.debug(f'[{row_nu}][{operacao_compra}][{open_time}] => Compra: {valor_compra:.4f}')
      comprado = True

    if comprado:
      diff = 100 * (_data.iloc[row_nu:row_nu + 1]['close'].values[0] - valor_compra) / valor_compra

    if (abs(diff) >= stop_loss) and comprado:
      valor_venda = _data.iloc[row_nu:row_nu + 1]['close'].values[0]
      if revert:
        if operacao_compra.startswith('SOBE'):
          saldo -= saldo * (diff / 100)
        else:
          saldo -= saldo * (-diff / 100)
      else:
        if operacao_compra.startswith('SOBE'):
          saldo += saldo * (diff / 100)
        else:
          saldo += saldo * (-diff / 100)

      log.debug(f'[{row_nu}][{operacao_compra}][{open_time}] => Venda: {valor_venda:.4f} => Diff: {diff:.2f}% ==> PnL: $ {saldo:.2f}')
      comprado = False
    # Fim simulação

  if operacao_compra == '':
    log.info('Nenhuma operação de Compra e Venda foi realizada!')

  log.info(f'>>>Saldo: {saldo}')
  return saldo


def validate_score_test_data(exp, final_model, label, test_data, ajusted_test_data):
  log.info('start_train_engine: predicting final model...')

  df_final_predict = exp.predict_model(final_model, data=ajusted_test_data, raw_score=False, verbose=False)

  res_score = None
  if test_data is not None:
    df_final_predict[label] = test_data[label]
    df_final_predict['_score'] = df_final_predict['prediction_label'] == df_final_predict[label]

    log.info(f'Score Mean: {df_final_predict["_score"].mean()}')
    log.info(f'Score Group: \n{df_final_predict[[label, "_score"]].groupby(label).mean()}')
    res_score = df_final_predict[[label, '_score']].groupby(label).mean().copy()

  return df_final_predict, res_score


def save_results(model_name,
                 symbol,
                 interval,
                 estimator,
                 imbalance_method,
                 train_size,
                 start_train_date,
                 start_test_date,
                 numeric_features,
                 regression_times,
                 regression_features,
                 times_regression_profit_and_loss,
                 stop_loss,
                 fold,
                 #start_value,
                 #final_value,
                 use_all_data_to_train,
                 no_tune,
                 res_score,
                 arguments):

  simulation_results_filename = f'{myenv.datadir}/resultado_simulacao_{symbol}_{interval}.csv'
  if (os.path.exists(simulation_results_filename)):
    df_resultado_simulacao = pd.read_csv(simulation_results_filename, sep=';')
  else:
    df_resultado_simulacao = pd.DataFrame()

  result_simulado = {}
  result_simulado['data'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  result_simulado['symbol'] = symbol
  result_simulado['interval'] = interval
  result_simulado['estimator'] = estimator
  result_simulado['imbalance_method'] = imbalance_method
  result_simulado['stop_loss'] = stop_loss
  result_simulado['regression_times'] = regression_times
  result_simulado['times_regression_profit_and_loss'] = times_regression_profit_and_loss
  if res_score is not None:
    for i in range(0, len(res_score.index.values)):
      field = res_score.index.values[i].split('_')[0]
      result_simulado[field] = round(res_score["_score"].values[i], 4)
    result_simulado['max_score'] = max(result_simulado['CAI'], result_simulado['SOBE'])

  #result_simulado['profit_and_loss_value'] = round(final_value - start_value, 2)
  #result_simulado['start_value'] = round(start_value, 2)
  #result_simulado['final_value'] = round(final_value, 2)
  result_simulado['numeric_features'] = numeric_features
  result_simulado['regression_features'] = regression_features
  result_simulado['train_size'] = train_size
  result_simulado['use-all-data-to-train'] = use_all_data_to_train
  result_simulado['start_train_date'] = start_train_date
  result_simulado['start_test_date'] = start_test_date
  result_simulado['fold'] = fold
  result_simulado['no-tune'] = no_tune

  result_simulado['model_name'] = model_name
  result_simulado['arguments'] = arguments

  df_resultado_simulacao = pd.concat([df_resultado_simulacao, pd.DataFrame([result_simulado])], ignore_index=True)
  df_resultado_simulacao.sort_values('max_score', inplace=True)

  df_resultado_simulacao.to_csv(simulation_results_filename, sep=';', index=False)

def has_results(symbol,
                interval,
                estimator,
                imbalance_method,                 
                numeric_features,
                times_regression_profit_and_loss,
                stop_loss):
  
  log.debug(f'symbol={symbol},interval={interval},estimator={estimator},imbalance_method={imbalance_method},numeric_features={numeric_features}\
,times_regression_profit_and_loss={times_regression_profit_and_loss},stop_loss={stop_loss}')
  
  log.debug(f'symbol={type(symbol)},interval={type(interval)},estimator={type(estimator)},imbalance_method={type(imbalance_method)}\
,numeric_features={type(numeric_features)},times_regression_profit_and_loss={type(times_regression_profit_and_loss)},stop_loss={type(stop_loss)}')


  simulation_results_filename = f'{myenv.datadir}/resultado_simulacao_{symbol}_{interval}.csv'
  log.debug(f'simulation_results_filename: {simulation_results_filename}') 
  if not os.path.exists(simulation_results_filename):
    return False

  df_resultado_simulacao = pd.read_csv(simulation_results_filename, sep=';')
  #df_resultado_simulacao.info()
  log.debug(f'df_resultado_simulacao.shape[0]: {df_resultado_simulacao.shape[0]}')
  if df_resultado_simulacao.shape[0] > 0:
    chave = (df_resultado_simulacao['symbol'] == symbol) & \
        (df_resultado_simulacao['interval'] == interval) & \
        (df_resultado_simulacao['estimator'] == estimator) & \
        (df_resultado_simulacao['imbalance_method'] == imbalance_method) & \
        (df_resultado_simulacao['stop_loss'] == float(stop_loss)) & \
        (df_resultado_simulacao['times_regression_profit_and_loss'] == int(times_regression_profit_and_loss)) & \
        (df_resultado_simulacao['numeric_features'] == numeric_features)
#        (df_resultado_simulacao['regression_times'] == regression_times) & \    
#        (df_resultado_simulacao['start_test_date'] == start_test_date) & \

    log.debug(f'chave.sum(): {chave.sum()}')
    if chave.sum() > 0:
      log.debug(f'{symbol}_{interval}_{estimator}: fix_imbalance_method: {imbalance_method} already exists')
      return True

  return False

def get_params_robo_trader(params, param_name, type = None, split = False):

  cai = params['CAI'][param_name] if 'CAI' in params else None
  sobe = params['SOBE'][param_name] if 'SOBE' in params else None
  
  if split:
    cai = cai.split(',') if cai is not None else None
    sobe = sobe.split(',') if sobe is not None else None
  
  result = {}
  match type:
    case 'int': 
      result['CAI'] = int(cai) if cai is not None else 0
      result['SOBE'] = int(sobe) if sobe is not None else 0
      return result
    case 'float': 
      result['CAI'] = float(cai) if cai is not None else 0.0
      result['SOBE'] = float(sobe) if sobe is not None else 0.0
      return result
    
  return {'CAI': cai, 'SOBE': sobe}