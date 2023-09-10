import sys
# sys.path.append('../')

from src.utils import *
from src.calcEMA import calc_RSI
from src.myenv import *
from src.train import *
import src.send_message as sm

from binance.client import Client
from pycaret.classification import ClassificationExperiment
from sklearn.model_selection import train_test_split

import pandas as pd
import datetime

import time
import traceback


def start_predict_engine(symbol,
                         estimator='xgboost',
                         tail=-1,
                         start_train_date='2010-01-01',
                         start_test_date=None,
                         numeric_features=myenv.data_numeric_fields,
                         stop_loss=myenv.stop_loss,
                         regression_times=myenv.regression_times,
                         times_regression_profit_and_loss=myenv.times_regression_profit_and_loss,
                         calc_rsi=True,
                         trace=False):
    print('start_predict_engine: regression_times: ', regression_times)

    use_cols = date_features + numeric_features
    print('start_predict_engine: use_cols: ', use_cols)

    experiment, model = load_model(symbol, estimator, stop_loss, regression_times, times_regression_profit_and_loss)  # cassification_experiment
    print('start_predict_engine: model loaded!')

    print('start_predict_engine: reagind all data...')
    df_database, _ = prepare_all_data(symbol,
                                      start_train_date,
                                      calc_rsi,
                                      numeric_features,
                                      False,
                                      regression_profit_and_loss,
                                      regression_times)

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
        max_date = get_max_date(df_database)
        open_time = df_database.tail(1)["open_time"].values[0]
        print('start_predict_engine: max_date: ', max_date)

        df_klines = get_klines(symbol, max_date=max_date.strftime('%Y-%m-%d'), adjust_index=True, limit=1, columns=use_cols)
        # print('start_predict_engine: df_klines.tail(1): \n', df_klines.tail(1))
        df_database = pd.concat([df_database, df_klines])
        df_database.drop_duplicates(keep='last', subset=['open_time'], inplace=True)
        df_database.sort_index(inplace=True)
        df_database['symbol'] = symbol
        df_database = parse_type_fields(df_database)
        print('start_predict_engine: df_database.shape: ', df_database.shape)
        # print('start_predict_engine: df_database.tail(1): \n', df_database.tail(1))

        print('start_predict_engine: calc_rsi...')
        if calc_rsi:
            df_database = calc_RSI(df_database, last_one=True)
            df_database.to_csv(f'log_after_{cont}_calc_RSI.csv', sep=';', index=False) if trace else None

        print(f'start_predict_engine: regression_times {regression_times}...')
        if regression_times > 0:
            df_database, _ = regresstion_times(df_database, numeric_features, regression_times, last_one=True)
            df_database.to_csv(f'log_after_{cont}_regresstion_times.csv', sep=';', index=False) if trace else None

        df_database[['open_time'] + numeric_features].to_csv('log_experiment_data.log', sep=';', index=False) if trace else None

        # Calculo compra e venda
        valor_atual = df_database.tail(1)["close"].values[0]
        print(f'start_predict_engine: valor_atual: {valor_atual}')

        if comprado:
            diff = 100 * (valor_atual - valor_compra) / valor_compra
            msg = f'*COMPRADO*: symbol: {symbol} - open_time: {open_time} - valor comprado: {valor_compra} - valor atual: {valor_atual} - Margem: {diff}% - Saldo: {saldo}'
            sm.send_status_to_telegram(msg)

        if (abs(diff) >= stop_loss) and comprado:
            if operacao_compra.startswith('SOBE'):
                saldo += round(saldo * (diff / 100), 2)
            else:
                saldo += round(saldo * (-diff / 100), 2)

            open_time = df_database.tail(1)["open_time"].values[0]
            comprado = False
            valor_compra = 0
            operacao_compra = ''

            msg = f'Venda: symbol: {symbol} - open_time: {open_time} - valor comprado: {valor_compra} - valor venda: {valor_atual} - Margem: {diff}% - Saldo: {saldo}'
            sm.send_to_telegram(msg)
        # Fim calculo compra e venda

        print('start_predict_engine: start predict_model...')
        df_predict = experiment.predict_model(model, df_database.tail(1), verbose=False)
        df_predict.to_csv(f'log_after_{cont}_df_predict.csv', sep=';', index=False) if trace else None
        # Inicio calculo compra
        operacao = df_predict['prediction_label'].values[0]
        print(f'start_predict_engine: operacao predita: {operacao}')
        if (operacao.startswith('SOBE') or operacao.startswith('CAI')) and not comprado:
            comprado = True
            valor_compra = df_predict.tail(1)["close"].values[0]
            operacao_compra = operacao
            rsi = df_predict.tail(1)["rsi"].values[0]

            msg = f'Compra: symbol: {symbol} - open_time: {open_time} - valor comprado: {valor_compra} - RSI: {rsi} - Saldo: {saldo}'
            sm.send_to_telegram(msg)
        # Fim calculo compra

        time.sleep(sleep_refresh)
        cont += 1
        cont_aviso += 1
        if cont_aviso > 1000:
            sm.send_status_to_telegram('Ainda trabalhando...')
            cont_aviso = 0


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
                    times_regression_profit_and_loss = float(arg.split('=')[1])

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

            sm.send_to_telegram(f'Iniciando Modelo Preditor para Symbol: {symbol}...')
            print(f'robo:main: args: {args}')
            print(f'robo:main: numeric_features: {numeric_features}')
            start_predict_engine(symbol, estimator, -1, start_train_date, start_test_date, numeric_features, stop_loss, regression_times,
                                 times_regression_profit_and_loss, calc_rsi)
        except Exception as e:
            sm.send_status_to_telegram('ERRO: ' + str(e))
            traceback.print_exc()
        finally:
            gc.collect()
            time.sleep(60)
            continue


if __name__ == '__main__':
    main(sys.argv[1:])
