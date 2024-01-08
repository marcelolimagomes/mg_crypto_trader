from binance import Client, helpers
from binance.exceptions import BinanceAPIException
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


def format_date(date):
    result = ''
    _format = '%Y-%m-%d %H:%M:%S'
    if date is not None:
        result = f'{date}'
        if isinstance(date, np.datetime64):
            result = pd.to_datetime(date, unit='ms').strftime(_format)
        elif isinstance(date, datetime):
            result = date.strftime(_format)

    return result


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


def status_order_limit(symbol, interval):
    id_limit = f'{symbol}_{interval}_limit'

    order = None
    res_is_purchased = False
    take_profit = 0.0
    try:
        order = get_client().get_order(symbol=symbol, origClientOrderId=id_limit)
        res_is_purchased = order['status'] in [Client.ORDER_STATUS_NEW, Client.ORDER_STATUS_PARTIALLY_FILLED]
        take_profit = float(order['price'])
    except BinanceAPIException as e:
        if e.code != -2013:
            log.exception(f'is_purchased - ERROR: {e}')
            sm.send_status_to_telegram(f'{symbol}_{interval} - is_purchased - ERROR: {e}')
    except Exception as e:
        log.exception(f'is_purchased - ERROR: {e}')
        sm.send_status_to_telegram(f'{symbol}_{interval} - is_purchased - ERROR: {e}')

    return res_is_purchased, order, take_profit


def status_order_stop(symbol, interval):
    id_stop = f'{symbol}_{interval}_stop'

    order = None
    res_is_purchased = False
    stop_loss = 0.0
    try:
        order = get_client().get_order(symbol=symbol, origClientOrderId=id_stop)
        res_is_purchased = order['status'] in [Client.ORDER_STATUS_NEW, Client.ORDER_STATUS_PARTIALLY_FILLED]
        stop_loss = float(order['stopPrice'])
    except Exception as e:
        log.exception(f'status_order_stop - ERROR: {e}')
        sm.send_status_to_telegram(f'{symbol}_{interval} - status_order_stop - ERROR: {e}')

    return res_is_purchased, order, stop_loss


def status_order_buy(symbol, interval):
    res_is_buying = False
    id = f'{symbol}_{interval}_buy'
    purchased_price = 0.0
    executed_qty = 0.0
    amount_invested = 0.0
    try:
        order = get_client().get_order(symbol=symbol, origClientOrderId=id)
        res_is_buying = order['status'] in [Client.ORDER_STATUS_NEW, Client.ORDER_STATUS_PARTIALLY_FILLED]
        purchased_price = float(order['price'])
        executed_qty = float(order['executedQty'])
        amount_invested = purchased_price * executed_qty
    except Exception as e:
        log.exception(f'status_order_buy - ERROR: {e}')
        sm.send_status_to_telegram(f'{symbol}_{interval} - status_order_buy - ERROR: {e}')
    return res_is_buying, order, purchased_price, executed_qty, amount_invested


def register_operation(params):
    status, order_buy_id, order_oco_id = None, None, None
    try:
        log.warn(f'Trying to register order_limit_buy: Params> {params}')

        symbol = params['symbol']
        interval = params['interval']
        new_client_order_id = f'{symbol}_{interval}_buy'
        quantity_precision = params['quantity_precision']
        price_precision = params['price_precision']
        amount_invested = params['amount_invested']

        price_order = round(params['purchase_price'], price_precision)  # get_client().get_symbol_ticker(symbol=params['symbol'])
        params['purchase_price'] = price_order
        quantity = round(amount_invested / price_order, quantity_precision)
        params['quantity'] = quantity

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
        is_buying, order_buy_id, _, _, _ = status_order_buy(params["symbol"], params["interval"])
        status = order_buy_id['status']
        while is_buying:
            if purchase_attemps > myenv.max_purchase_attemps:
                if status == Client.ORDER_STATUS_NEW:  # Can't buy after max_purchase_attemps, than cancel
                    get_client().cancel_order(symbol=params['symbol'], origClientOrderId=new_client_order_id)
                    err_msg = f'Can\'t buy {params["symbol"]} after {myenv.max_purchase_attemps} attemps'
                    log.error(err_msg)
                    sm.send_status_to_telegram(f'[ERROR]: {symbol}_{interval}: {err_msg}')
                    return status, None, None
                elif status == Client.ORDER_STATUS_PARTIALLY_FILLED:  # Partially filled, than try sell quantity partially filled
                    msg = f'BUYING OrderId: {order_buy_id["orderId"]} Partially filled, than try sell quantity partially filled'
                    log.warn(msg)
                    sm.send_status_to_telegram(f'[WARNING]: {symbol}_{interval}: {msg}')
                    break
            purchase_attemps += 1
            time.sleep(1)
            is_buying, order_buy_id, _, _, _ = status_order_buy(params["symbol"], params["interval"])
            status = order_buy_id['status']

        order_oco_id = register_oco_sell(params)
    except Exception as e:
        log.exception(e)
        sm.send_status_to_telegram(f'[ERROR] register_operation: {symbol}_{interval}: {e}')
        traceback.print_stack()

    return status, order_buy_id, order_oco_id


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

    _quantity = get_asset_balance(symbol.split('USDT')[0], quantity_precision)
    if _quantity > round(params['quantity'] * 1.05, quantity_precision):  # Sell 5% more quantity if total quantity of asset is more than
        quantity = round(params['quantity'] * 1.05, quantity_precision)
    elif _quantity >= round(params['quantity'], quantity_precision):  # Sell exactly quantity buyed
        quantity = round(params['quantity'], quantity_precision)
    else:
        quantity = _quantity  # Sell all quantity of asset

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
    # sm.send_to_telegram(info_msg)

    oder_oco_sell_id = get_client().order_oco_sell(**oco_params)
    sm.send_status_to_telegram(info_msg + f' - oder_oco_sell_id: {oder_oco_sell_id}')
    log.warn(f'oder_oco_sell_id: {oder_oco_sell_id}')
    return oder_oco_sell_id


def parse_kline_from_stream(df_stream_kline: pd.DataFrame, maintain_cols=['open_time', 'close', 'symbol']):
    rename = {
     't': 'open_time',  # Kline start time
     'T': 'close_time',  # Kline close time
     's': 'symbol',  # Symbol
     'i': 'interval',  # Interval
     'f': 'first_trade_id',  # First trade ID
     'L': 'last_trade_id',  # Last trade ID
     'o': 'open',  # Open price
     'c': 'close',  # Close price
     'h': 'high',  # High price
     'l': 'low',  # Low price
     'v': 'base_asset_volume',    # Base asset volume
     'n': 'number_of_trades',       # Number of trades
     'x': 'is_closed',     # Is this kline closed?
     'q': 'quote_asset_volume',  # Quote asset volume
     'V': 'taker_buy_base_asset_volume',     # Taker buy base asset volume
     'Q': 'taker_buy_quote_asset_volume',   # Taker buy quote asset volume
     'B': 'ignore'   # Ignore
        }
    df_stream_kline.rename(columns=rename, inplace=True, errors='ignore')
    del_cols = []
    for key in df_stream_kline.columns:
        if key not in maintain_cols:
            del_cols.append(key)

    df_stream_kline.drop(columns=del_cols, inplace=True, errors='ignore')
    return df_stream_kline


def to_periods(delta: datetime, interval='1h'):
    days = delta.days
    hours = delta.seconds // 3600
    minutes = delta.seconds % 3600 // 60
    total_minutes = (days * 24 * 60) + (hours * 60) + minutes + (delta.seconds // 60)

    resutl = 0
    match(interval):
        case '1m':
            resutl = total_minutes
        case '5m':
            resutl = total_minutes // 5
        case '15m':
            resutl = total_minutes // 15
        case '30m':
            resutl = total_minutes // 30
        case '1h':
            resutl = total_minutes // 60
    return resutl - 1


def get_latest_update(data: pd.DataFrame) -> datetime:
    return pd.to_datetime(data.tail(1)['open_time'].values[0])


def has_to_update(data: pd.DataFrame, interval: str):
    shape = data.shape[0]
    latest_periods = 0
    if shape > 2:
        utcnow = datetime.utcnow()
        data = data.iloc[shape - 2:shape - 1]  # ignore last row
        latest_update = get_latest_update(data)
        delta_update = utcnow - latest_update
        latest_periods = to_periods(delta_update, interval)
        log.debug(f'has_to_update: interval: {interval} - utcnow: {utcnow} - latest_update: {latest_update} - delta_update: {delta_update} - latest_periods: {latest_periods}')
    return latest_periods
