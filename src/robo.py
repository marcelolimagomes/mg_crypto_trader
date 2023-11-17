import sys
# sys.path.append('../')

from src.utils import *
from src.calcEMA import calc_RSI
from src.myenv import *
from train_ml import *
import src.send_message as sm

from binance.client import Client
from pycaret.classification import ClassificationExperiment
from sklearn.model_selection import train_test_split

import pandas as pd
import datetime

import time
import traceback

import numpy as np


def start_predict_engine(symbol,
                         interval,
                         estimator='xgboost',
                         tail=-1,
                         start_train_date='2010-01-01',
                         start_test_date=None,
                         numeric_features=myenv.data_numeric_fields,
                         stop_loss=myenv.stop_loss,
                         regression_times=myenv.regression_times,
                         times_regression_profit_and_loss=myenv.times_regression_profit_and_loss,
                         calc_rsi=True,
                         saldo=myenv.saldo_inicial,
                         verbose=False):
  use_cols = date_features + numeric_features
  print(f'start_predict_engine: Now: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - \nParameters: ')
  print(f' --> symbol: {symbol}')
  print(f' --> estimator: {estimator}')
  print(f' --> regression_times: {regression_times}')
  print(f' --> tail: {tail}')
  print(f' --> start_train_date: {start_train_date}')
  print(f' --> start_test_date: {start_test_date}')
  print(f' --> numeric_features: {numeric_features}')
  print(f' --> stop_loss: {stop_loss}')
  print(f' --> regression_times: {regression_times}')
  print(f' --> times_regression_profit_and_loss: {times_regression_profit_and_loss}')
  print(f' --> calc_rsi: {calc_rsi}')
  print(f' --> saldo: {saldo}')
  print(f' --> verbose: {verbose}')
  print(f' --> use_cols: {use_cols}')

  model_name_init = get_model_name_to_load(symbol, interval, estimator, stop_loss, regression_times, times_regression_profit_and_loss)
  experiment, model = load_model(symbol, interval, estimator, stop_loss, regression_times, times_regression_profit_and_loss)  # cassification_experiment
  print(f'start_predict_engine: model {model_name_init} loaded.')

  df_database, _ = prepare_all_data(symbol,
                                    start_train_date,
                                    calc_rsi,
                                    numeric_features,
                                    False,
                                    times_regression_profit_and_loss,
                                    regression_times,
                                    start_train_date is None,
                                    stop_loss,
                                    verbose,
                                    )
  print(f'start_predict_engine: df_database.shape: {df_database.shape} - start_train_date: {start_train_date}')

  cont = 0
  cont_aviso = 0
  operacao = ''
  operacao_compra = ''
  comprado = False
  valor_compra = 0
  valor_atual = 0
  diff = 0.0
  print('start_predict_engine: starting loop monitoring...')
  while True:
    print('------------------------>>')
    print(f'start_predict_engine: Loop  -->  Symbol: {symbol} - Cont: {cont} - Now: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    try:
      model_name = get_model_name_to_load(symbol, interval, estimator, stop_loss, regression_times, times_regression_profit_and_loss)
      if model_name != model_name_init:
        experiment, model = load_model(symbol, interval, estimator, stop_loss, regression_times, times_regression_profit_and_loss)  # cassification_experiment
        model_name_init = model_name
        sm.send_status_to_telegram(f'start_predict_engine: reload new model. New model name: {model_name} - Old model name: {model_name_init}')

      max_date = get_max_date(df_database)
      open_time = df_database.tail(1)["open_time"].dt.strftime('%Y-%m-%d %H:%M:%S').values[0]
      print('start_predict_engine: max_date: ', max_date)

      df_klines = get_klines(symbol, max_date=max_date.strftime('%Y-%m-%d'), limit=1, columns=use_cols)
      df_database = pd.concat([df_database, df_klines])
      df_database.drop_duplicates(keep='last', subset=['open_time'], inplace=True)
      df_database.sort_index(inplace=True)
      df_database['symbol'] = symbol
      df_database = parse_type_fields(df_database)
      print('start_predict_engine: Updated data - df_database.shape: ', df_database.shape)

      print('start_predict_engine: calc_rsi...')
      if calc_rsi:
        df_database = calc_RSI(df_database)  # , last_one=True)

      print(f'start_predict_engine: regression_times {regression_times}...')
      if regression_times > 0:
        df_database, _ = regresstion_times(df_database, numeric_features, regression_times, last_one=True)

      # Calculo compra e venda
      valor_atual = df_database.tail(1)["close"].values[0]
      print(f'start_predict_engine: valor_atual: >> $ {valor_atual:.4f} <<')

      if comprado:
        diff = 100 * (valor_atual - valor_compra) / valor_compra

      if (abs(diff) >= stop_loss) and comprado:
        if operacao_compra.startswith('SOBE'):
          saldo += saldo * (diff / 100)
        else:
          saldo += saldo * (-diff / 100)
        msg = f'Venda: Symbol: {symbol} - open_time: {open_time} - Operação: {operacao_compra} - Valor Comprado: {valor_compra:.4f} - Valor Venda: {valor_atual:.4f} - Variação: {diff:.4f}% - PnL: $ {saldo:.2f}'
        sm.send_to_telegram(msg)
        # Reset variaveis
        comprado = False
        valor_compra = 0
        operacao_compra = ''
      # Fim calculo compra e venda

      if not comprado:
        print('start_predict_engine: start predict_model...')
        df_predict = experiment.predict_model(model, df_database.tail(1), verbose=verbose)
        # Inicio calculo compra
        operacao = df_predict['prediction_label'].values[0]
        print(f'start_predict_engine: operacao predita: {operacao}')
        if (operacao.startswith('SOBE') or operacao.startswith('CAI')):
          comprado = True
          valor_compra = df_predict.tail(1)["close"].values[0]
          operacao_compra = operacao
          rsi = df_predict.tail(1)["rsi"].values[0]

          msg = f'Compra: Symbol: {symbol} - open_time: {open_time} - Operação: {operacao_compra} - Valor Comprado: {valor_compra:.4f} - RSI: {rsi:.2f} - PnL: $ {saldo:.2f}'
          sm.send_to_telegram(msg)
        # Fim calculo compra
      gc.collect()
    except Exception as e:
      traceback.print_exc()
      sm.send_status_to_telegram('ERROR: ' + str(e))
      gc.collect()
    finally:
      time.sleep(sleep_refresh)
      cont += 1
      cont_aviso += 1
      if cont_aviso > 100:
        cont_aviso = 0
        if comprado:
          msg = f'*COMPRADO*: Symbol: {symbol} - open_time: {open_time} - Operação: {operacao_compra} - Valor Comprado: {valor_compra:.4f} - Valor Atual: {valor_atual:.4f} - Variação: {diff:.4f}% - PnL: $ {saldo:.2f}'
          sm.send_status_to_telegram(msg)
        else:
          msg = f'*NÃO COMPRADO*: Symbol: {symbol} - open_time: {open_time} - Valor Atual: {valor_atual:.4f} - PnL: $ {saldo:.2f}'
          sm.send_status_to_telegram(msg)


def main(args):
  while True:
    try:
      symbol = myenv.symbol
      estimator = myenv.estimator
      stop_loss = myenv.stop_loss
      regression_times = myenv.regression_times
      times_regression_profit_and_loss = myenv.times_regression_profit_and_loss
      calc_rsi = False
      numeric_features = myenv.data_numeric_fields
      start_train_date = '2010-01-01'
      start_test_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
      saldo_inicial = myenv.saldo_inicial
      verbose = False

      for arg in args:
        if (arg.startswith('-download_data')):
          sm.send_to_telegram('Iniciando MG Crypto Trader...')
          sm.send_to_telegram('Atualizando base de dados')
          download_data()
          sm.send_to_telegram('Base atualizada')

      for arg in args:
        if (arg.startswith('-symbol=')):
          symbol = arg.split('=')[1]

        if (arg.startswith('-estimator=')):
          estimator = arg.split('=')[1]

        if (arg.startswith('-stop-loss=')):
          stop_loss = float(arg.split('=')[1])

        if (arg.startswith('-regression-times=')):
          regression_times = int(arg.split('=')[1])

        if (arg.startswith('-regression-profit-and-loss=')):
          times_regression_profit_and_loss = int(arg.split('=')[1])

        if (arg.startswith('-calc-rsi')):
          calc_rsi = True

        if (arg.startswith('-numeric-features=')):
          aux = arg.split('=')[1]
          numeric_features = aux.split(',')

        if (arg.startswith('-all-cols')):
          aux = float_kline_cols + integer_kline_cols  # + ['close_time']
          numeric_features = aux

        if (arg.startswith('-start-train-date=')):
          start_train_date = arg.split('=')[1]

        if (arg.startswith('-start-test-date=')):
          start_test_date = arg.split('=')[1]

        if (arg.startswith('-saldo-inicial=')):
          saldo_inicial = float(arg.split('=')[1])

        if (arg.startswith('-verbose')):
          verbose = True
          os.environ["PYCARET_CUSTOM_LOGGING_LEVEL"] = "DEBUG"

      sm.send_status_to_telegram(f'bot:main: Iniciando Modelo Preditor para Symbol: {symbol} - Args: {args}')
      print(f'bot:main: args: {args}')
      print(f'bot:main: numeric_features: {numeric_features}')
      start_predict_engine(symbol, estimator, -1, start_train_date, start_test_date, numeric_features, stop_loss, regression_times,
                           times_regression_profit_and_loss, calc_rsi, saldo_inicial, verbose)
    except Exception as e:
      traceback.print_exc()
      sm.send_status_to_telegram('ERRO: ' + str(e))
    finally:
      gc.collect()
      time.sleep(60)


if __name__ == '__main__':
  main(sys.argv[1:])
