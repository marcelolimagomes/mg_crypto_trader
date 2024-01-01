from binance import ThreadedWebsocketManager, helpers
from datetime import datetime

import src.utils_binance as utils
import src.myenv as myenv
import src.send_message as sm
import pandas as pd
import numpy as np
import time
import src.calcEMA as calc_utils
import os
import logging
import threading
import traceback

# global
balance: float = 0.0


class RoboTraderIndex():
    def __init__(self, params: dict):
        self._params = params
        self._all_data: pd.DataFrame = params['all_data']
        self._all_cols = list(params['all_data'].columns)

        # self._twm = params['twm']

        # Single arguments
        self._symbol = params['symbol']
        self._interval = params['interval']
        self._mutex = threading.Lock()
        self._latest_update: datetime = params['latest_update']

        # List arguments
        self._start_date = params['start_date']
        self._target_margin = float(params['target_margin'])
        self._stop_loss_multiplier = int(params['stop_loss_multiplier'])
        self._p_ema = int(params['p_ema'])
        self._min_rsi = int(params['min_rsi'])
        self._max_rsi = int(params['max_rsi'])

        # Boolean arguments
        self._calc_rsi = params['calc_rsi']
        self._verbose = params['verbose']

        # Initialize logging
        if 'log_level' in params:
            self.log = self._configure_log(params['log_level'])
        else:
            self.log = self._configure_log(myenv.log_level)

        self.log.debug(f'ROBO TRADER >>>>>>> {params}')

        self.ix = f'{self._symbol}_{self._interval}'
        sm.send_status_to_telegram(f'Starting Robo Trader for {self._symbol}_{self._interval}')

    def _configure_log(self, log_level):
        log_file_path = os.path.join(myenv.logdir, f'robo_trader_{self._symbol}_{self._interval}.log')
        logger = logging.getLogger(f'robo_trader_{self._symbol}_{self._interval}')
        logger.propagate = False

        logger.setLevel(log_level)

        fh = logging.FileHandler(log_file_path, mode='a', delay=True)
        fh.setFormatter(logging.Formatter(f'[%(asctime)s] - %(levelname)s - RoboTrader: {self._symbol}-{self._interval} - %(message)s', '%Y-%m-%d %H:%M:%S'))
        fh.setLevel(log_level)

        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(f'[%(asctime)s] - %(levelname)s - RoboTrader: {self._symbol}-{self._interval} - %(message)s', '%Y-%m-%d %H:%M:%S'))
        sh.setLevel(log_level)

        logger.addHandler(fh)
        logger.addHandler(sh)
        return logger

    def log_info(self, purchased: bool, open_time, purchase_price: float, actual_price: float, margin_operation: float, amount_invested: float, profit_and_loss: float, balance: float, take_profit: float,
                 stop_loss: float, target_margin: float, strategy: str, p_ema_label: str, p_ema_value: float):
        '''
        _open_time = open_time
        if open_time is not None:
            if isinstance(open_time, np.datetime64):
                _open_time = pd.to_datetime(open_time, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
            else:
                _open_time = open_time.strftime('%Y-%m-%d %H:%M:%S')
        '''

        if purchased:
            msg = f'[BNC]*PURCHASED*: {self._symbol}_{self._interval} {strategy} TM: {target_margin:.2f}% '
            msg += f'PP: ${purchase_price:.6f} AP: ${actual_price:.6f} MO: {100*margin_operation:.2f}% AI: ${amount_invested:.2f} '
            msg += f'PnL: ${profit_and_loss:.2f} TP: ${take_profit:.6f} SL: ${stop_loss:.6f} B: ${balance:.2f} {p_ema_label}: ${p_ema_value:.6f}'
        else:
            msg = f'[BNC]<NOT PURCHASED>: {self._symbol}_{self._interval} AP: ${actual_price:.6f} B: ${balance:.2f} TM: {target_margin:.2f}% {p_ema_label}: ${p_ema_value:.6f}'
        self.log.info(f'{msg}')
        sm.send_status_to_telegram(msg)

    def log_buy(self, open_time, strategy, purchase_price, amount_invested, balance, target_margin, take_profit, stop_loss, rsi, restored=False):
        _open_time = open_time
        if open_time is not None:
            if isinstance(open_time, np.datetime64):
                _open_time = pd.to_datetime(open_time, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
            else:
                _open_time = open_time.strftime('%Y-%m-%d %H:%M:%S')

        status = ''
        if restored:
            status = '-RESTORED'

        msg = f'[BINANCE]*BUYING{status}*: {self._symbol}_{self._interval} - OT: {_open_time} - St: {strategy} - TM: {target_margin:.2f}% - '
        msg += f'PP: $ {purchase_price:.6f} - AI: $ {amount_invested:.2f} - TP: $ {take_profit:.6f} - '
        msg += f'SL: $ {stop_loss:.6f} - RSI: {rsi:.2f} - B: $ {balance:.2f}'
        self.log.info(f'{msg}')
        sm.send_to_telegram(msg)

    def log_selling(self, open_time, strategy, purchase_price, actual_price, margin_operation, amount_invested, profit_and_loss, balance, take_profit,
                    stop_loss, target_margin, rsi):
        _open_time = open_time
        if open_time is not None:
            if isinstance(open_time, np.datetime64):
                _open_time = pd.to_datetime(open_time, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
            else:
                _open_time = open_time.strftime('%Y-%m-%d %H:%M:%S')

        sum_pnl = utils.get_sum_pnl()
        msg = f'[BINANCE]*SELLING*: {self._symbol}_{self._interval} - OT: {_open_time} - St: {strategy} - TM: {target_margin:.2f}% '
        msg += f'- PP: $ {purchase_price:.6f} - AP: $ {actual_price:.6f} - MO: {100*margin_operation:.2f}% - AI: $ {amount_invested:.2f} '
        msg += f'- TP: $ {take_profit:.6f} - SL: $ {stop_loss:.6f} - RSI: {rsi:.2f} - B: $ {balance:.2f} - PnL O: $ {profit_and_loss:.2f} - Sum PnL: $ {sum_pnl:.2f}'
        self.log.info(f'{msg}')
        sm.send_to_telegram(msg)

    def validate_short_or_long(self, strategy):
        return strategy.startswith('LONG') or strategy.startswith('SHORT')

    def is_long(self, strategy):
        return strategy.startswith('LONG')

    def is_short(self, strategy):
        return strategy.startswith('SHORT')

    def _data_preprocessing(self):
        utcnow = datetime.utcnow()
        delta_update = utcnow - self._latest_update
        latest_periods = utils.to_periods(delta_update, self._interval)

        if latest_periods > 0:
            self.log.info(f'Updating data: utcnow: {utcnow} - latest_update: {self._latest_update} - delta_update: {delta_update} - latest_periods: {latest_periods}.')
            self.update_data_from_web(limit=latest_periods + 100)

    def _feature_engineering(self):
        return

    def feature_engineering_on_loop(self):
        self.log.debug(f'Calculating EMA\'s for key {self._symbol}_{self._interval}...')
        rsi = 0.0
        p_ema_value = 0.0

        try:
            p_ema_label = f'ema_{self._p_ema}p'
            self.log.debug(f'Start Calculating {p_ema_label} and RSI...')
            self._mutex.acquire()
            self._all_data = calc_utils.calc_ema_periods(self._all_data, periods_of_time=[self._p_ema], diff_price=False)
            self._all_data = calc_utils.calc_RSI(self._all_data)
            self._mutex.release()
            self.log.debug(f'After Calculating {p_ema_label} and RSI - all_data.shape: {self._all_data.shape}')

            rsi = float(self._all_data.tail(1)['rsi'].values[0])
            p_ema_value = float(self._all_data.tail(1)[p_ema_label].values[0])
        except Exception as e:
            self.log.error(f'Error on feature_engineering_on_loop: {e}')
            if self._mutex.locked():
                self._mutex.release()

        return rsi, p_ema_value

    def update_data_from_web(self, limit=1000):
        df_klines = utils.get_klines(symbol=self._symbol, interval=self._interval, max_date=None, limit=limit, columns=self._all_cols, parse_dates=True)

        self.log.debug(f'df_klines shape: {df_klines.shape}')
        df_klines.info() if self._verbose else None

        try:
            self._mutex.acquire()
            self._all_data = pd.concat([self._all_data, df_klines])
            self._all_data.drop_duplicates(keep='last', subset=['open_time'], inplace=True)
            self._all_data.sort_index(inplace=True)
            self._mutex.release()
            self.log.debug(f'update_data_from_web: Updated - all_data.shape: {self._all_data.shape}')
            self._all_data.info() if self._verbose else None
        except Exception as e:
            self.log.error(f'Error on update_data_from_web: {e}')
            if self._mutex.locked():
                self._mutex.release()

        self.log.debug(f'Updated - all_data.shape: {self._all_data.shape}')

    def update_data(self):
        now_price = self._all_data.tail(1)['close'].values[0]
        latest_closed_candle_open_time = self._all_data.iloc[self._all_data.shape[0] - 2:self._all_data.shape[0] - 1]['open_time'].values[0]

        self._all_data.info() if self._verbose else None
        self.log.debug(f'update_data: df_klines.shape: {self._all_data.shape}')

        self.log.debug(f'update_data: price: ${now_price} - latest_closed_candle_open_time: {latest_closed_candle_open_time}')
        return now_price, latest_closed_candle_open_time

    def handle_socket_kline(self, msg):
        try:
            self.log.debug(f'handle_socket_kline:{msg}')
            df_klines = utils.parse_kline_from_stream(pd.DataFrame(data=[msg['k']]), maintain_cols=self._all_cols)
            df_klines = utils.parse_type_fields(df_klines, parse_dates=True)
            df_klines = utils.adjust_index(df_klines)
            df_klines.info() if self._verbose else None
            self.log.debug(f'handle_socket_kline: df_klines.shape: {df_klines.shape}')

            self._mutex.acquire()
            self._all_data = pd.concat([self._all_data, df_klines])
            self._all_data.drop_duplicates(keep='last', subset=['open_time'], inplace=True)
            self._all_data.sort_index(inplace=True)
            self._mutex.release()
            self.log.debug(f'handle_socket_kline: Updated - all_data.shape: {self._all_data.shape}')
        except Exception as e:
            self.log.error(f'***ERROR*** handle_socket_kline: {e}')
            sm.send_status_to_telegram(f'{self.ix} ***ERROR*** handle_socket_kline: {e}')
            if self._mutex.locked():
                self._mutex.release()

    def run(self):
        self.log.info(f'Start _data_preprocessing...')
        self._data_preprocessing()
        self.log.info(f'Start _feature_engineering...')
        self._feature_engineering()

        twm = ThreadedWebsocketManager()
        twm.start()
        self.log.info(f'ThreadedWebsocketManager: {twm.start_kline_socket(callback=self.handle_socket_kline, symbol=self._symbol, interval=self._interval)}')

        symbol_info, symbol_precision, quote_precision, quantity_precision, price_precision, step_size, tick_size = utils.get_symbol_info(self._symbol)

        cont = 0
        cont_aviso = 101

        purchased = False
        purchase_price = 1.0
        actual_price = 0.0
        amount_invested = 0.0
        amount_to_invest = 0.0
        take_profit = 0.0
        stop_loss = 0.0
        profit_and_loss = 0.0
        strategy = ''
        margin_operation = 0.0
        target_margin = 0.0
        rsi = 0.0
        p_ema_label = f'ema_{self._p_ema}p'
        p_ema_value = 0.0

        global balance
        balance = utils.get_account_balance()  # ok
        target_margin = self._target_margin

        no_ammount_to_invest_count = 0
        while True:
            try:
                # Update data
                latest_periods = utils.has_to_update(self._all_data, self._interval)
                if latest_periods > 2:
                    self.log.warn(f'>> REFRESH data from web. Periods: {latest_periods}.')
                    sm.send_status_to_telegram(f'<<{self.ix}>> REFRESH data from web. Periods: {latest_periods}.')
                    self.update_data_from_web(latest_periods + 5)

                actual_price, open_time = self.update_data()
                rsi, p_ema_value = self.feature_engineering_on_loop()
                strategy = utils.predict_strategy_index(self._all_data, self._p_ema, self._max_rsi, self._min_rsi)  # ok
                self.log.info(f'Predicted Strategy: {strategy} - min_rsi: {self._min_rsi:.2f}% - max_rsi: {self._max_rsi:.2f}%')
                if self.is_long(strategy):  # Olny BUY with LONG strategy. If true, BUY
                    purchased, order_sell_limit = utils.is_purchased(self._symbol, self._interval)
                    self.log.info(f'Purchased: {purchased} - Price: {actual_price:.{symbol_precision}f} - Target Margin: {target_margin:.2f}% - RSI: {rsi:.2f}% - {p_ema_label}: ${p_ema_value:.{symbol_precision}f} - Balance: ${balance:.{quote_precision}f}')
                    if not purchased:
                        amount_to_invest, balance = utils.get_amount_to_invest()  # ok
                        if amount_to_invest > myenv.min_amount_to_invest:
                            take_profit, stop_loss = utils.calc_take_profit_stop_loss(strategy, actual_price, target_margin, self._stop_loss_multiplier)  # ok
                            ledger_params = utils.get_params_operation(open_time, self._symbol, self._interval, 'BUY', target_margin, amount_to_invest,
                                                                       take_profit, stop_loss, actual_price, rsi, 0.0, 0.0, 0.0, strategy, balance,
                                                                       symbol_precision, quote_precision, quantity_precision, price_precision, step_size, tick_size)  # ok
                            status_buy, order_buy_id, order_sell_id = utils.register_operation(ledger_params)
                            purchase_price = actual_price
                            amount_invested = amount_to_invest
                            if order_buy_id is not None:
                                msg = f'{self.ix}-{strategy}: *ORDER BUY* - Status: {status_buy} AP: ${actual_price:.{symbol_precision}f} PP: ${purchase_price:.{symbol_precision}f} AI: ${amount_to_invest:.2f} '
                                msg += f'TP: ${take_profit:.{symbol_precision}f} SL: ${stop_loss:.{symbol_precision}f} {p_ema_label}: ${p_ema_value:.{symbol_precision}f} '
                                msg += f'TM: {target_margin:.2f}% RSI: {rsi:.2f}% B: ${balance:.{quote_precision}f} SELL: {"OK" if order_sell_id is not None else "ERROR"} '
                                sm.send_to_telegram(msg)
                                self.log.debug(msg)
                        else:
                            msg = f'No Amount to invest: ${balance:.{quote_precision}f} Min: ${myenv.min_amount_to_invest:.{quote_precision}f} '
                            self.log.warn(msg)
                            no_ammount_to_invest_count += 1

                cont_aviso += 1
                if cont_aviso > 50:  # send status to telegram each x loop
                    cont_aviso = 0
                    # purchased, purchase_price, amount_invested, take_profit, stop_loss = utils.is_purchased(self._symbol, self._interval)
                    if purchased:
                        margin_operation = (actual_price - purchase_price) / purchase_price
                        profit_and_loss = amount_invested * margin_operation
                    self.log_info(purchased, open_time, purchase_price, actual_price, margin_operation, amount_invested, profit_and_loss, balance,
                                  take_profit, stop_loss, target_margin, strategy, p_ema_label, p_ema_value)
                    if no_ammount_to_invest_count > 0:
                        msg = f'No Amount to invest :Tryed {no_ammount_to_invest_count} times.  ${balance:.{quote_precision}f} Min: ${myenv.min_amount_to_invest:.{quote_precision}f} '
                        sm.send_status_to_telegram(f'{self.ix}: {msg}')
                        no_ammount_to_invest_count = 0

            except Exception as e:
                traceback.print_stack()
                err_msg = f'ERROR: symbol: {self._symbol} - interval: {self._interval} - Exception: {e}'
                self.log.exception(err_msg)
                sm.send_status_to_telegram(err_msg)
            finally:
                # Sleep in each loop
                time.sleep(myenv.sleep_refresh)
