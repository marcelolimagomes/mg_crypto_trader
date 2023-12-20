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
client: Client = None


def get_client() -> Client:
    global client
    if client is None:
        client = login_binance()
        log.debug(f'New Instance of Client: {client}')
        print(f'New Instance of Client: {client}')

    return client


def login_binance() -> Client:
    with open(f'{sys.path[0]}/mg.key', 'r') as file:
        first_line = file.readline()
        if not first_line:
            raise Exception('mg.key is empty')
        key = first_line.split('##$$')[0]
        sec = first_line.split('##$$')[1]

    _client = Client(key, sec)
    return _client


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
    klines = get_client().get_historical_klines(symbol=symbol, interval=interval, start_str=max_date, limit=limit)

    columns = remove_cols_for_klines(columns)

    # log.info('get_klines: columns: ', columns)
    df_klines = pd.DataFrame(data=klines, columns=myenv.all_klines_cols)[columns]
    df_klines = parse_type_fields(df_klines, parse_dates=parse_dates)
    df_klines = adjust_index(df_klines)
    delta = datetime.now() - start_time
    # Print the delta time in days, hours, minutes, and seconds
    log.debug(f'get_klines: shape: {df_klines.shape} - Delta time: {delta.seconds % 60} seconds')
    return df_klines


def get_params_operation(operation_date, symbol: str, interval: str, operation: str, target_margin: float, amount_invested: float, take_profit: float, stop_loss: float, purchase_price: float, rsi: float, sell_price: float,
                         profit_and_loss: float, margin_operation: float, strategy: str, balance: float, symbol_precision: int, quote_precision: int, quantity_precision: int, price_precision: int, step_size: float, tick_size: float):
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
                        'step_size': step_size,
                        'quantity_precision': quantity_precision,
                        'quote_precision': quote_precision,
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


def calc_take_profit_stop_loss(strategy, actual_value: float, margin: float, stop_loss_multiplier=myenv.stop_loss_multiplier):
    take_profit_value = 0.0
    stop_loss_value = 0.0
    if strategy.startswith('SHORT'):  # Short
        take_profit_value = actual_value * (1 - margin / 100)
        stop_loss_value = actual_value * (1 + (margin * stop_loss_multiplier) / 100)
    elif strategy.startswith('LONG'):  # Long
        take_profit_value = actual_value * (1 + margin / 100)
        stop_loss_value = actual_value * (1 - (margin * stop_loss_multiplier) / 100)
    return take_profit_value, stop_loss_value


def get_precision(tick_size: float) -> int:
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


def get_symbol_info(symbol):
    '''
        return symbol_info, symbol_precision, step_size, tick_size
    '''
    symbol_info = get_client().get_symbol_info(symbol=symbol)
    symbol_precision = int(symbol_info['baseAssetPrecision'])
    quote_precision = int(symbol_info['quoteAssetPrecision'])
    for filter in symbol_info['filters']:
        if filter['filterType'] == 'LOT_SIZE':
            # stepSize defines the intervals that a quantity/icebergQty can be increased/decreased by
            step_size = float(filter['stepSize'])
        if filter['filterType'] == 'PRICE_FILTER':
            # tickSize defines the intervals that a price/stopPrice can be increased/decreased by; disabled on tickSize == 0
            tick_size = float(filter['tickSize'])

    quantity_precision = get_precision(step_size)
    price_precision = get_precision(tick_size)

    symbol_info['step_size'] = step_size
    symbol_info['quantity_precision'] = quantity_precision
    symbol_info['tick_size'] = tick_size
    symbol_info['price_precision'] = price_precision
    symbol_info['symbol_precision'] = symbol_precision

    return symbol_info, symbol_precision, quote_precision, quantity_precision, price_precision, step_size, tick_size


def get_account_balance(asset=myenv.asset_balance_currency):
    asset_balance = get_client().get_asset_balance(asset)
    balance = float(asset_balance['free'])

    return balance


def get_amount_to_invest():
    balance = get_account_balance()
    amount_invested = 0.0

    if balance >= myenv.min_amount_to_invest:
        if balance >= myenv.default_amount_to_invest:
            amount_invested = myenv.default_amount_to_invest
        elif balance > 0 and balance < myenv.default_amount_to_invest:
            amount_invested = balance
            # balance -= amount_invested

    return amount_invested, balance


def is_purchased(symbol, interval):
    id_buy = f'{symbol}_{interval}_buy'
    id_limit = f'{symbol}_{interval}_limit'
    id_stop = f'{symbol}_{interval}_stop'
    orders = get_client().get_all_orders(symbol=symbol)

    res_is_purchased = False
    purchased_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    executed_qty = 0.0
    amount_invested = 0.0
    try:
        df_order = pd.DataFrame(orders)
        if df_order.shape[0] > 0:
            key = (df_order['clientOrderId'] == id_buy) | (df_order['clientOrderId'] == id_limit) | (df_order['clientOrderId'] == id_stop)
            if key.sum() > 0:
                df_order = df_order[key]
                res_is_purchased = df_order['status'].isin([Client.ORDER_STATUS_NEW, Client.ORDER_STATUS_PARTIALLY_FILLED, Client.ORDER_STATUS_PENDING_CANCEL]).sum() > 0
                if res_is_purchased:
                    has_buy = df_order['clientOrderId'] == id_buy
                    if has_buy.sum() > 0:
                        purchased_price = float(df_order[has_buy].tail(1)['price'].values[0])
                        executed_qty = float(df_order[has_buy].tail(1)['executedQty'].values[0])
                        amount_invested = executed_qty * purchased_price
                    has_limit = df_order['clientOrderId'] == id_limit
                    if has_limit.sum() > 0:
                        take_profit = float(df_order[has_limit].tail(1)['price'].values[0])
                    has_stop = df_order['clientOrderId'] == id_stop
                    if has_stop.sum() > 0:
                        stop_loss = float(df_order[has_stop].tail(1)['stopPrice'].values[0])
    except Exception as e:
        log.exception(e)
        sm.send_status_to_telegram(f'ERROR on call is_purchased({symbol}, {interval}): {e}')
    return res_is_purchased, purchased_price, amount_invested, take_profit, stop_loss


def is_buying(symbol, interval):
    id = f'{symbol}_{interval}_buy'
    orders = get_client().get_all_orders(symbol=symbol)
    for order in orders:
        if order['clientOrderId'] == id:
            if (order['side'] == Client.SIDE_BUY) and \
                    (order['status'] in [Client.ORDER_STATUS_NEW, Client.ORDER_STATUS_PARTIALLY_FILLED, Client.ORDER_STATUS_PENDING_CANCEL]):
                return True
    return False


def register_operation(params):
    log.warn(f'Trying to register order_limit_buy: Params> {params}')

    symbol = params['symbol']
    interval = params['interval']
    new_client_order_id = f'{symbol}_{interval}_buy'
    quantity_precision = params['quantity_precision']
    price_precision = params['price_precision']
    amount_invested = params['amount_invested']

    price_order = round(params['purchase_price'], price_precision)  # get_client().get_symbol_ticker(symbol=params['symbol'])
    quantity = round(amount_invested / price_order, quantity_precision)

    order_params = {}
    order_params['symbol'] = symbol
    order_params['quantity'] = quantity
    order_params['price'] = str(price_order)
    order_params['newClientOrderId'] = new_client_order_id

    order_buy_id = get_client().order_limit_buy(**order_params)

    info_msg = f'ORDER BUY: {order_params}'
    log.warn(info_msg)
    sm.send_status_to_telegram(info_msg + f'order_buy_id: {order_buy_id}')
    log.warn(f'order_buy_id: {order_buy_id}')

    purchase_attemps = 0
    while is_buying(params["symbol"], params["interval"]):
        if purchase_attemps > myenv.max_purchase_attemps:
            get_client()._delete('openOrders', True, data={'symbol': params['symbol']})
            err_msg = f'Can\'t buy {params["symbol"]} after {myenv.max_purchase_attemps} attemps'
            log.error(err_msg)
            sm.send_status_to_telegram(err_msg)
            return None, None
        purchase_attemps += 1
        time.sleep(1)

    order_oco_id = register_oco_sell(params)

    return order_buy_id, order_oco_id


def get_asset_balance(asset=myenv.asset_balance_currency, quantity_precision: int = 2):
    filled_asset_balance = get_client().get_asset_balance(asset)
    int_quantity = filled_asset_balance['free'].split('.')[0]
    frac_quantity = filled_asset_balance['free'].split('.')[1][:quantity_precision]
    quantity = float(int_quantity + '.' + frac_quantity)
    return quantity


def register_oco_sell(params):
    log.warn(f'Trying to register order_oco_sell: Params> {params}')

    symbol = params['symbol']
    interval = params['interval']

    limit_client_order_id = f'{symbol}_{interval}_limit'
    stop_client_order_id = f'{symbol}_{interval}_stop'
    price_precision = params['price_precision']
    quantity_precision = params['quantity_precision']
    take_profit = round(params['take_profit'], price_precision)
    stop_loss_trigger = round(params['stop_loss'], price_precision)
    stop_loss_target = round(stop_loss_trigger * 0.95, price_precision)

    quantity = get_asset_balance(symbol.split('USDT')[0], quantity_precision)

    oco_params = {}
    oco_params['symbol'] = params['symbol']
    oco_params['quantity'] = quantity
    oco_params['price'] = str(take_profit)
    oco_params['stopPrice'] = str(stop_loss_trigger)
    oco_params['stopLimitPrice'] = str(stop_loss_target)
    oco_params['stopLimitTimeInForce'] = 'GTC'
    oco_params['limitClientOrderId'] = limit_client_order_id
    oco_params['stopClientOrderId'] = stop_client_order_id

    info_msg = f'ORDER SELL: {symbol}_{interval} - oco_params: {oco_params} - price_precision: {price_precision} - quantity_precision: {quantity_precision}'
    log.warn(info_msg)
    sm.send_to_telegram(info_msg)

    oder_oco_sell_id = get_client().order_oco_sell(**oco_params)
    sm.send_status_to_telegram(info_msg + f' - oder_oco_sell_id: {oder_oco_sell_id}')
    log.warn(f'oder_oco_sell_id: {oder_oco_sell_id}')
    return oder_oco_sell_id
