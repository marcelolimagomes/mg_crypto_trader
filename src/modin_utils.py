import src.myenv as myenv

import os
import logging
import traceback
from datetime import datetime, timedelta

import modin.pandas as pd
import dpnp as np
import numpy

from binance.client import Client

log = logging.getLogger()


# VARS

base_dir = '/home/marcelo/des/mg_crypto_trader'
datadir = base_dir + '/data'
logdir = base_dir + '/logs'


def get_max_date(df_database, start_date='2010-01-01'):
    max_date = datetime.strptime(start_date, '%Y-%m-%d')
    if df_database is not None and df_database.shape[0] > 0:
        max_date = pd.to_datetime(df_database['open_time'].max(), unit='ms')
    return max_date


def truncate_data_in_days(data, days):
    min_date = data['open_time'].min()
    max_date = data['open_time'].max()
    print(f'Data Loaded: Min date: {min_date} - Max date: {max_date}')
    validate_start_data = max_date - numpy.timedelta64(days, 'D')
    data = data[data['open_time'] >= validate_start_data]

    min_date = data['open_time'].min()
    max_date = data['open_time'].max()
    print(f'All Data Filtered: Min Date: {min_date} - Max_date: {max_date} - Shape: {data.shape}')

    return data


def date_parser(x):
    return pd.to_datetime(x, unit='ms')


def parse_type_fields(df, parse_dates=False):
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
    return df


def adjust_index(df):
    df.drop_duplicates(keep='last', subset=['open_time'], inplace=True)
    df.index = df['open_time']
    df.index.name = 'ix_open_time'
    df.sort_index(inplace=True)
    return df


def get_database_name(symbol, interval):
    return f'{datadir}/{symbol}/{symbol}_{interval}.dat'


def get_database(symbol, interval='1h', tail=-1, columns=['open_time', 'close'], parse_dates=True):
    database_name = get_database_name(symbol, interval)
    print(f'get_database: name: {database_name}')

    df_database = pd.DataFrame()
    print(f'get_database: columns: {columns}')
    if os.path.exists(database_name):
        if parse_dates:
            df_database = pd.read_csv(database_name, sep=';', parse_dates=myenv.date_features, date_parser=date_parser, decimal='.', usecols=columns, )
        else:
            df_database = pd.read_csv(database_name, sep=';', decimal='.', usecols=columns, )

        df_database.info()
        df_database = parse_type_fields(df_database, parse_dates)
        print('df_database + parse_type_fields', df_database)
        df_database = adjust_index(df_database)
        df_database = df_database[columns]
    if tail > 0:
        df_database = df_database.tail(tail)
    print(f'get_database: count_rows: {df_database.shape[0]} - symbol: {symbol}_{interval} - tail: {tail}')
    print(f'get_database: duplicated: {df_database.index.duplicated().sum()}')
    return df_database


def get_data(symbol, save_database=False, interval='1h', tail=-1, columns=['open_time', 'close'], parse_dates=True, updata_data_from_web=True, start_date='2010-01-01'):
    database_name = get_database_name(symbol, interval)
    print(f'get_data: Loading database: {database_name}')
    df_database = get_database(symbol=symbol, interval=interval, tail=tail, columns=columns, parse_dates=parse_dates)
    print(f'Shape database on disk: {df_database.shape}')

    print(f'Filtering start date: {start_date}')
    if parse_dates:
        df_database = df_database[df_database['open_time'] >= start_date]
        print(f'New shape after filtering start date. Shape: {df_database.shape}')

    max_date = get_max_date(df_database, start_date=start_date)
    max_date_aux = ''
    new_data = False
    if updata_data_from_web:
        print(f'get_data: Downloading data for symbol: {symbol} - max_date: {max_date}')
        while (max_date != max_date_aux):
            new_data = True
            print(f'get_data: max_date: {max_date} - max_date_aux: {max_date_aux}')
            max_date_aux = get_max_date(df_database, start_date=start_date)
            print(f'get_data: Max date database: {max_date_aux}')

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
        df_database.to_csv(database_name, sep=';', index=False, )
        print(f'get_data: Database updated at {database_name}')

    print(f'New shape after get_data: {df_database.shape}')
    if tail > 0:
        df_database = df_database.tail(tail)
    return df_database


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

    # print('get_klines: columns: ', columns)
    df_klines = pd.DataFrame(data=klines, columns=myenv.all_klines_cols)[columns]
    df_klines = parse_type_fields(df_klines, parse_dates=parse_dates)
    df_klines = adjust_index(df_klines)
    delta = datetime.now() - start_time
    # Print the delta time in days, hours, minutes, and seconds
    print(f'get_klines: shape: {df_klines.shape} - Delta time: {delta.seconds % 60} seconds')
    return df_klines


def finalize_index_train(train_param, df_result_simulation_list, count=1):
    result = 'SUCESS'
    pnl = simule_index_trading(
        train_param['all_data'],
        train_param['symbol'],
        train_param['interval'],
        train_param['p_ema'],
        myenv.default_amount_invested,
        train_param['target_margin'],
        train_param['min_rsi'],
        train_param['max_rsi'],
        train_param['stop_loss_multiplier'])

    ix_symbol = f'{train_param["symbol"]}_{train_param["interval"]}'
    save = (count % 1000 == 0)  # Save results every 5000 iterations

    # lock.acquire()  # Lock Thread to register results
    df_result_simulation_list[ix_symbol] = save_index_results(
        df_result_simulation_list[ix_symbol],
        train_param['symbol'],
        train_param['interval'],
        train_param['target_margin'],
        train_param['p_ema'],
        train_param['min_rsi'],
        train_param['max_rsi'],
        myenv.default_amount_invested,
        pnl,
        train_param['stop_loss_multiplier'],
        train_param['arguments'],
        False)
    if save:
        save_all_results_index_simulation(df_result_simulation_list)
        print(f'_finalize_index_train: Count==> {count}: Partial Results Save for all symbols')
        print(f'_finalize_index_train: Count==> {count}: Partial Results Save for all symbols')
        # lock.release()
    return result


def only_save_index_results(df_result_simulation, symbol, interval):
    simulation_results_filename = f'{myenv.datadir}/resultado_simulacao_index_{symbol}_{interval}.csv'
    df_result_simulation.sort_values('pnl', inplace=True)
    df_result_simulation.to_csv(simulation_results_filename, sep=';', index=False)


def save_all_results_index_simulation(df_result_simulation_list):
    for key in df_result_simulation_list.keys():
        print(f'KEY>>>>>>>>>>> {key}')
        symbol = key.split('_')[0]
        interval = key.split('_')[1]
        if df_result_simulation_list[key].shape[0] > 0:
            only_save_index_results(df_result_simulation_list[key], symbol, interval)
            print(f'save_all_results: Results Save for => {key}')
            print(f'save_all_results: Results Save for => {key}')


def get_index_results(symbol, interval):
    df_resultado_simulacao = None
    simulation_results_filename = f'{myenv.datadir}/resultado_simulacao_index_{symbol}_{interval}.csv'
    if (os.path.exists(simulation_results_filename)):
        df_resultado_simulacao = pd.read_csv(simulation_results_filename, sep=';')
    else:
        df_resultado_simulacao = pd.DataFrame()
    return df_resultado_simulacao


def save_index_results(df_result_simulation, symbol, interval, target_margin, p_ema, min_rsi, max_rsi, amount_invested, pnl, stop_loss_multiplier, arguments, save_result=False):
    if df_result_simulation is None:
        df_result_simulation = get_index_results(symbol, interval)

    result_simulation = {}
    result_simulation['data'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_simulation['symbol'] = symbol
    result_simulation['interval'] = interval
    result_simulation['target_margin'] = float(target_margin)
    result_simulation['p_ema'] = int(p_ema)
    result_simulation['min_rsi'] = int(min_rsi)
    result_simulation['max_rsi'] = int(max_rsi)
    result_simulation['amount_invested'] = round(float(amount_invested), 2)
    result_simulation['stop_loss_multiplier'] = int(stop_loss_multiplier)
    result_simulation['pnl'] = round(float(pnl), 2)
    result_simulation['arguments'] = arguments

    new_df_result_simulation = pd.concat([df_result_simulation, pd.DataFrame([result_simulation])], ignore_index=True)

    if save_result:
        simulation_results_filename = f'{myenv.datadir}/resultado_simulacao_index_{symbol}_{interval}.csv'
        new_df_result_simulation.sort_values('pnl', inplace=True)
        new_df_result_simulation.to_csv(simulation_results_filename, sep=';', index=False)

    return new_df_result_simulation


def calc_take_profit_stop_loss(strategy, actual_value, margin, stop_loss_multiplier=myenv.stop_loss_multiplier):
    take_profit_value = 0.0
    stop_loss_value = 0.0
    if strategy.startswith('SHORT'):  # Short
        take_profit_value = actual_value * (1 - margin / 100)
        stop_loss_value = actual_value * (1 + (margin * stop_loss_multiplier) / 100)
    elif strategy.startswith('LONG'):  # Long
        take_profit_value = actual_value * (1 + margin / 100)
        stop_loss_value = actual_value * (1 - (margin * stop_loss_multiplier) / 100)
    return take_profit_value, stop_loss_value


def predict_strategy_index(all_data: pd.DataFrame, p_ema=myenv.p_ema, max_rsi=myenv.max_rsi, min_rsi=myenv.min_rsi):
    result_strategy = 'ESTAVEL'

    price = 0.0
    rsi = 0.0
    p_ema_value = 0.0

    if type(all_data) == pd.DataFrame:
        price = all_data.tail(1)['close'].values[0]
        rsi = all_data.tail(1)['rsi'].values[0]
        p_ema_value = all_data.tail(1)[f'ema_{p_ema}p'].values[0]
    elif type(all_data) == pd.Series:
        price = all_data['close']
        rsi = all_data['rsi']
        p_ema_value = all_data[f'ema_{p_ema}p']

    if price > p_ema_value and rsi >= max_rsi:
        result_strategy = 'SHORT'

    if price < p_ema_value and rsi <= min_rsi:
        result_strategy = 'LONG'

    return result_strategy


def simule_index_trading(_data: pd.DataFrame, symbol, interval, p_ema: int, start_amount_invested=100, target_margin=myenv.stop_loss, min_rsi=myenv.min_rsi, max_rsi=myenv.max_rsi, stop_loss_multiplier=myenv.stop_loss_multiplier):
    purchased = False
    perform_sell = False
    purchase_price = 0.0
    purchase_strategy = ''
    print(f'Start Simule Trading: {symbol}_{interval} - Shape: {_data.shape} - Amount Invested: {start_amount_invested:.2f} - Target Margin: {target_margin}% - EMA: {p_ema}p - Min RSI: {min_rsi}% - Max RSI: {max_rsi}% - Stop Loss Multiplier: {stop_loss_multiplier}')

    for _, row in _data.iterrows():
        actual_price = row['close']
        rsi = row['rsi']
        p_ema_value = row[f'ema_{p_ema}p']
        open_time = row['open_time']
        # strategy = row['strategy']
        # take_profit = row['take_profit']
        # stop_loss = row['stop_loss']

        if isinstance(open_time, numpy.datetime64):
            _open_time = pd.to_datetime(open_time, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
        else:
            _open_time = open_time.strftime('%Y-%m-%d %H:%M:%S')

        if not purchased:
            strategy = predict_strategy_index(row, p_ema, max_rsi, min_rsi)
            # print(f'[{row_nu}][{strategy}] => Purchased: {purchased} - Price: {actual_price:.6f} - RSI: {rsi:.2f} - ema_{p_ema}p: {p_ema_value:.2f} - Min RSI: {min_rsi} - Max RSI: {max_rsi}')
            if strategy.startswith('LONG') or strategy.startswith('SHORT'):  # If true, BUY
                purchased = True
                purchase_price = actual_price
                purchase_strategy = strategy
                take_profit, stop_loss = calc_take_profit_stop_loss(strategy, purchase_price, target_margin, stop_loss_multiplier)
                print(f'BUY[{_open_time}]: {purchase_strategy} - Price: {actual_price:.6f} - Target Margin: {target_margin:.2f}% - Take Profit: \
{take_profit:.2f} - Stop Loss: {stop_loss:.2f} - RSI: {rsi:.2f}% - ema_{p_ema}p: {p_ema_value:.2f} - Min RSI: {min_rsi}% - Max RSI: {max_rsi}% - Stop Loss Multiplier: {stop_loss_multiplier}')
                continue

        if purchased:  # and (operation.startswith('LONG') or operation.startswith('SHORT')):
            # print(f'[{row_nu}][{strategy}] => Purchased: {purchased} - Price: {actual_price:.6f} - RSI: {rsi:.2f} - ema_{p_ema}p: {p_ema_value:.2f} - Min RSI: {min_rsi} - Max RSI: {max_rsi}')
            if purchase_strategy.startswith('LONG'):
                margin_operation = (actual_price - purchase_price) / purchase_price
                if ((actual_price >= take_profit) or (actual_price <= stop_loss)):  # Long ==> Sell - Take Profit / Stop Loss
                    perform_sell = True
            elif purchase_strategy.startswith('SHORT'):
                margin_operation = (purchase_price - actual_price) / purchase_price
                if ((actual_price <= take_profit) or (actual_price >= stop_loss)):  # Short ==> Sell - Take Profit / Stop Loss
                    perform_sell = True

            if perform_sell:
                start_amount_invested = (1 + margin_operation) * start_amount_invested
                print(f'SELL[{_open_time}]: {purchase_strategy} - Price: {actual_price:.6f} - Purchase Price: {purchase_price:.6f} - Target Margin: {target_margin:.2f}% - Margin Operation:\
{margin_operation*100:.2f}% - Take Profit: {take_profit:.2f} - Stop Loss: {stop_loss:.2f} - RSI: {rsi:.2f} - ema_{p_ema}p: {p_ema_value:.2f} - Sum PnL: {start_amount_invested:.2f}  - Stop Loss Multiplier: {stop_loss_multiplier}')
                perform_sell = False
                purchase_strategy = ''
                purchased = False
    print(f'{symbol}_{interval} - Result Simule Trading: {start_amount_invested:.2f}')

    return start_amount_invested
