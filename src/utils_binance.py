from binance import Client
from datetime import datetime, timedelta

import sys
import traceback
import logging
import pandas as pd
import numpy as np
import time

import src.myenv as myenv
import src.send_message as sm

log = logging.getLogger()


def login_binance() -> Client:
    with open(f'{sys.path[0]}/mg.key', 'r') as file:
        first_line = file.readline()
        if not first_line:
            raise Exception('mg.key is empty')
        key = first_line.split('##$$')[0]
        sec = first_line.split('##$$')[1]

    client = Client(key, sec)
    return client


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
        traceback.print_stack()

    return df


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


def get_params_operation(operation_date, symbol, interval, operation, target_margin, amount_invested, take_profit, stop_loss, purchase_price, rsi, sell_price,
                         profit_and_loss, margin_operation, strategy, balance, symbol_precision, price_precision, tick_size, step_size):
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
                        'balance': balance,
                        'symbol_precision': symbol_precision,
                        'price_precision': price_precision,
                        'tick_size': tick_size,
                        'step_size': step_size
                        }
    return params_operation


def predict_strategy_index(all_data: pd.DataFrame, p_ema=myenv.p_ema, max_rsi=myenv.max_rsi, min_rsi=myenv.min_rsi):
    result_strategy = 'ESTAVEL'

    price = 0.0
    rsi = 0.0
    p_ema_value = 0.0

    if type(all_data) == pd.DataFrame:
        price = all_data.tail(1)['close'].values[0]
        rsi = all_data.tail(1)['rsi'].values[0]
        p_ema_value = all_data.tail(1)[f'ema_{p_ema}p'].values[0]
    elif type(all_data) == pd.core.series.Series:
        price = all_data['close']
        rsi = all_data['rsi']
        p_ema_value = all_data[f'ema_{p_ema}p']

    if price > p_ema_value and rsi >= max_rsi:
        result_strategy = 'SHORT'

    if price < p_ema_value and rsi <= min_rsi:
        result_strategy = 'LONG'

    return result_strategy


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


def get_precision(tick_size: float):
    result = 8
    if tick_size >= 1.0:
        result = 0
    elif tick_size >= 0.1:
        result = 1
    elif tick_size >= 0.01:
        result = 2
    elif tick_size >= 0.001:
        result = 3
    elif tick_size >= 0.0001:
        result = 4
    elif tick_size >= 0.00001:
        result = 5
    elif tick_size >= 0.000001:
        result = 6
    elif tick_size >= 0.0000001:
        result = 7
    return result


def get_symbol_info(client: Client, symbol):
    '''
        return symbol_info, symbol_precision, step_size, tick_size
    '''
    symbol_info = client.get_symbol_info(symbol=symbol)
    symbol_precision = int(symbol_info['baseAssetPrecision'])
    for filter in symbol_info['filters']:
        if filter['filterType'] == 'LOT_SIZE':
            step_size = float(filter['stepSize'])
        if filter['filterType'] == 'PRICE_FILTER':
            tick_size = float(filter['tickSize'])

    symbol_info['mg_step_size'] = step_size
    symbol_info['mg_symbol_precision'] = symbol_precision
    symbol_info['mg_tick_size'] = tick_size

    return symbol_info, symbol_precision, get_precision(tick_size), step_size, tick_size


def get_account_balance(client: Client, asset=myenv.asset_balance_currency):
    asset_balance = client.get_asset_balance(asset)
    balance = float(asset_balance['free'])

    return balance


def get_amount_to_invest(client: Client):
    balance = get_account_balance(client)
    amount_invested = 0.0

    if balance < 5.00:
        log.warning(f'Not enough balance: {balance:.2f} USDT')
    elif balance >= myenv.default_amount_invested:
        amount_invested = myenv.default_amount_invested
    elif balance > 0 and balance < myenv.default_amount_invested:
        amount_invested = balance
        balance -= amount_invested

    return amount_invested, balance


def is_purchased(client: Client, symbol):
    orders = client.get_all_orders(symbol=symbol)
    for order in orders:
        # (order['side'] == Client.SIDE_SELL) and \
        if (order['status'] in [Client.ORDER_STATUS_NEW, Client.ORDER_STATUS_PARTIALLY_FILLED, Client.ORDER_STATUS_PENDING_CANCEL]):
            return True
    return False


def is_buying(client: Client, symbol):
    orders = client.get_all_orders(symbol=symbol)
    for order in orders:
        if (order['side'] == Client.SIDE_BUY) and \
                (order['status'] in [Client.ORDER_STATUS_NEW, Client.ORDER_STATUS_PARTIALLY_FILLED, Client.ORDER_STATUS_PENDING_CANCEL]):
            return True
    return False


def register_operation(client: Client, params):
    price_order = client.get_symbol_ticker(symbol=params['symbol'])

    price = round(float(price_order['price']), params['symbol_precision'])
    params['purchase_price'] = price
    amount = params['amount_invested']
    quantity_precision = get_precision(params['step_size'])
    quantity = round(amount / price, quantity_precision)
    order_buy_id = None
    order_buy_id = client.order_limit_buy(symbol=params['symbol'], quantity=quantity, price=str(price))
    info_msg = f'ORDER BUY: symbol: {params["symbol"]} - {price_order} - quantity: {quantity}'
    log.info(info_msg)
    sm.send_to_telegram(info_msg)
    log.info(f'order_buy_id: {order_buy_id}')

    purchase_attemps = 0
    while is_buying(client, params['symbol']):
        if purchase_attemps > myenv.max_purchase_attemps:
            client._delete('openOrders', True, data={'symbol': params['symbol']})
            err_msg = f'Can\'t buy {params["symbol"]} after {myenv.max_purchase_attemps} attemps'
            log.error(err_msg)
            sm.send_status_to_telegram(err_msg)
            return None
        purchase_attemps += 1
        time.sleep(1)

    order_oco_id = register_oco_sell(client, params)

    return order_buy_id, order_oco_id


def register_oco_sell(client: Client, params):
    price_precision = get_precision(float(params['price_precision']))
    take_profit = round(float(params['take_profit']), price_precision)
    stop_loss_target = round(float(params['stop_loss']), price_precision)
    stop_loss_trigger = round(stop_loss_target * 1.05, price_precision)
    purchase_price = params['purchase_price']

    filled_asset_balance = client.get_asset_balance(params['symbol'].split('USDT')[0])
    int_quantity = filled_asset_balance['free'].split('.')[0]
    frac_quantity = filled_asset_balance['free'].split('.')[1][:price_precision + 1]
    quantity = float(int_quantity + '.' + frac_quantity)

    oder_oco_sell_id = None
    oder_oco_sell_id = client.order_oco_sell(symbol=params['symbol'], quantity=quantity, price=str(take_profit), stopPrice=str(stop_loss_trigger), stopLimitPrice=str(stop_loss_target), stopLimitTimeInForce='GTC')

    info_msg = f'ORDER SELL: symbol: {params["symbol"]} purchase_price: {purchase_price} price_precision: {price_precision} quantity: {quantity} take_profit: {take_profit} stop_loss_target: {stop_loss_target} stop_loss_trigger: {stop_loss_trigger}'
    log.info(info_msg)
    sm.send_to_telegram(info_msg)

    log.info(f'oder_oco_sell_id: {oder_oco_sell_id}')
    return oder_oco_sell_id
