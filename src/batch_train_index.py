from src.train_index import TrainIndex

import src.utils as utils
import src.calcEMA as calc_utils
import src.myenv as myenv
import src.send_message as sm
import logging
import pandas as pd
import datetime
import numpy as np
import sys
import threading
from multiprocessing import Process, Lock, Pool, Manager
import time
import os
import traceback


class BatchTrainIndex:
    def __init__(self,
                 update_data_from_web,
                 calc_rsi,
                 verbose,
                 log_level,
                 symbol_list,
                 interval_list,
                 target_margin_list,
                 min_rsi,
                 max_rsi,
                 range_p_ema,
                 retrain,
                 no_validate_duplicates
                 ):

        # Boolean arguments
        self._update_data_from_web = update_data_from_web
        self._calc_rsi = calc_rsi
        self._verbose = verbose
        # Single arguments
        self._log_level = log_level
        # List arguments
        self._symbol_list = symbol_list
        self._interval_list = interval_list
        self._target_margin_list = target_margin_list
        self._min_rsi = min_rsi
        self._max_rsi = max_rsi
        self._range_p_ema = range_p_ema
        self._params_list = []
        self._symbol_interval_list = []

        self._range_p_ema_ini = int(range_p_ema[0])
        self._range_p_ema_end = int(range_p_ema[1])

        self._retrain = retrain
        self._no_validate_duplicates = no_validate_duplicates

        # Private arguments
        self._all_data_list = {}

        # Initialize logging
        self.log = logging.getLogger("training_logger")

        # Prefix for log
        self.pl = f'BatchTrainIndex: '

        self._lock = None
        self._df_result_simulation = {}

    def _calc_indexes(self, ix_symbol):
        try:
            self.log.info(f'{self.pl}: Calculating RSI for symbol: {ix_symbol}...')
            self._all_data_list[ix_symbol] = calc_utils.calc_RSI(self._all_data_list[ix_symbol])

            self.log.info(f'{self.pl}: Calculating EMA\'s for key {ix_symbol}...')
            self._all_data_list[ix_symbol] = calc_utils.calc_ema_periods(self._all_data_list[ix_symbol], periods_of_time=[i for i in range(self._range_p_ema_ini, self._range_p_ema_end + 1, 25)], diff_price=False)

            self.log.info(f'{self.pl}: Info after calculating RSI and EMA\'s: ') if self._verbose else None
            self._all_data_list[ix_symbol] = utils.truncate_data_in_days(self._all_data_list[ix_symbol], myenv.days_to_validate_train)
            self._all_data_list[ix_symbol].dropna(inplace=True)
            self._all_data_list[ix_symbol].info() if self._verbose else None
        except Exception as e:
            self.log.exception(e)
            traceback.print_stack()

    def _data_collection(self):
        self.log.info(f'{self.pl}: Start Data Collection')

        if self._retrain:
            self._params_list, self._symbol_interval_list, self._range_p_ema_ini, self._range_p_ema_end = utils.prepare_best_params_index_retrain()
            self.log.info(f'{self.pl}: Loading data to memory: Symbol Interval List: {self._symbol_interval_list}')
            for symbol_interval in self._symbol_interval_list:
                symbol = symbol_interval['symbol']
                interval = symbol_interval['interval']
                ix_symbol = f'{symbol}_{interval}'
                try:
                    self.log.info(f'{self.pl}: Loading data for symbol: {ix_symbol}...')
                    _aux_data = utils.get_data(
                        symbol=symbol,
                        interval=interval,
                        columns=myenv.all_index_cols,
                        parse_dates=True,
                        updata_data_from_web=self._update_data_from_web)

                    self.log.info(f'{self.pl}: Store data in memory for symbol: {ix_symbol}...')
                    if _aux_data.shape[0] == 0:
                        raise Exception(f'Data for symbol: {ix_symbol} is empty')
                    self._all_data_list[ix_symbol] = _aux_data
                    self.log.info(f'{self.pl}: Loaded data to memory for symbol: {ix_symbol} - shape: {_aux_data.shape}')
                except Exception as e:
                    self.log.exception(e)
                    traceback.print_stack()
        else:
            self.log.info(f'{self.pl}: Loading data to memory: Symbols: {self._symbol_list} - Intervals: {self._interval_list}')
            for interval in self._interval_list:
                for symbol in self._symbol_list:
                    try:
                        ix_symbol = f'{symbol}_{interval}'
                        self.log.info(f'{self.pl}: Loading data for symbol: {ix_symbol}...')
                        _aux_data = utils.get_data(
                            symbol=symbol,
                            interval=interval,
                            columns=myenv.all_index_cols,
                            parse_dates=True,
                            updata_data_from_web=self._update_data_from_web)

                        _aux_data.info() if self._verbose else None
                        self.log.info(f'{self.pl}: Store data in memory for symbol: {ix_symbol}...')
                        if _aux_data.shape[0] == 0:
                            raise Exception(f'Data for symbol: {ix_symbol} is empty')
                        self._all_data_list[ix_symbol] = _aux_data
                        self.log.info(f'{self.pl}: Loaded data to memory for symbol: {ix_symbol} - shape: {_aux_data.shape}')
                    except Exception as e:
                        self.log.exception(e)
                        traceback.print_stack()

    def _data_preprocessing(self):
        self.log.info(f'{self.pl}: Start Data Preprocessing...')

        if self._retrain:
            for params in self._params_list:
                ix_symbol = f'{params["symbol"]}_{params["interval"]}'
                params['all_data'] = self._all_data_list[ix_symbol]

            for symbol_interval in self._symbol_interval_list:
                symbol = symbol_interval['symbol']
                interval = symbol_interval['interval']
                ix_symbol = f'{symbol}_{interval}'
                self._calc_indexes(ix_symbol)
        else:
            for interval in self._interval_list:
                for symbol in self._symbol_list:
                    ix_symbol = f'{symbol}_{interval}'
                    self._calc_indexes(ix_symbol)

    def prepare_param_to_train(self, df_result_simulation_list):
        params_list = []
        for interval in self._interval_list:
            for symbol in self._symbol_list:
                ix_symbol = f'{symbol}_{interval}'
                df_result_simulation_list[ix_symbol] = utils.get_index_results(symbol, interval)
                for target_margin in self._target_margin_list:
                    for p_ema in range(self._range_p_ema_ini, self._range_p_ema_end + 1, 25):
                        for min_rsi in range(self._min_rsi, 36 + 1, 2):
                            for max_rsi in range(72, self._max_rsi + 1, 2):
                                for stop_loss_multiplier in range(2, myenv.stop_loss_range_multiplier + 1):
                                    train_param = {
                                        'all_data': self._all_data_list[ix_symbol],
                                        'symbol': symbol,
                                        'interval': interval,
                                        'target_margin': float(target_margin),
                                        'min_rsi': int(min_rsi),
                                        'max_rsi': int(max_rsi),
                                        'p_ema': int(p_ema),
                                        'stop_loss_multiplier': int(stop_loss_multiplier),
                                        'calc_rsi': self._calc_rsi,
                                        'verbose': self._verbose,
                                        'lock': self._lock,
                                        'arguments': str(sys.argv[1:])}
                                    params_list.append(train_param)
                                    
        return params_list

    def run(self):
        self._data_collection()
        self._data_preprocessing()
        self.log.info(f'{self.pl}: Start Running...')

        with Pool(processes=(os.cpu_count() * 2)) as pool:
            manager = Manager()
            df_result_simulation_list = manager.dict()
            lock = manager.Lock()

            if self._retrain:
                for symbol_interval in self._symbol_interval_list:
                    symbol = symbol_interval['symbol']
                    interval = symbol_interval['interval']
                    ix_symbol = f'{symbol}_{interval}'
                    df_result_simulation_list[ix_symbol] = utils.get_index_results(symbol, interval)
            else:
                self._params_list = self.prepare_param_to_train(df_result_simulation_list)


            self.log.info(f'{self.pl}: Total Trainning Models: {len(self._params_list)}')
            self.log.info(f'{self.pl}: Will Start {len(self._params_list)} Threads.')
            self._params_list = sorted(self._params_list, key=lambda p: (str(p['p_ema']), str(p['target_margin']), str(p['max_rsi']), str(p['min_rsi']), str(p['symbol']), str(p['interval'])))
            process_list = []
            count = 0
            for p in self._params_list:
                ix_symbol = f"{p['symbol']}_{p['interval']}"
                _key_tm = f"{ix_symbol}_{p['target_margin']}"
                name = f"{_key_tm}_{p['p_ema']}_{p['min_rsi']}_{p['max_rsi']}_{p['stop_loss_multiplier']}"
                try:
                    if self._retrain or self._no_validate_duplicates:
                        process = pool.apply_async(func=utils.finalize_index_train, kwds={'train_param': p, 'lock': lock, 'df_result_simulation_list': df_result_simulation_list, 'count': count})
                        process_list.append({'name': name, 'symbol': p['symbol'], 'interval': p['interval'], 'target_margin': p['target_margin'], 'process': process})
                        count += 1
                    else:
                        if not utils.has_index_results(df_result_simulation_list[ix_symbol], p['symbol'], p['interval'], p['target_margin'], p['p_ema'], p['min_rsi'], p['max_rsi'], p['stop_loss_multiplier']):
                            process = pool.apply_async(func=utils.finalize_index_train, kwds={'train_param': p, 'lock': lock, 'df_result_simulation_list': df_result_simulation_list, 'count': count})
                            process_list.append({'name': name, 'symbol': p['symbol'], 'interval': p['interval'], 'target_margin': p['target_margin'], 'process': process})
                            count += 1
                except Exception as e:
                    self.log.exception(e)
                    traceback.print_stack()

            self.log.info(f'{self.pl}: Will Start collecting results for {len(self._params_list)} Threads.')
            results = []
            for p in process_list:
                try:
                    res = p['process'].get()
                    results.append(res)
                except Exception as e:
                    self.log.exception(e)
                    traceback.print_stack()
                    results.append("TIMEOUT_ERROR")

            self.log.info(f'{self.pl}: Saving Results for all symbols...')
            utils.save_all_results_index_simulation(df_result_simulation_list)
            self.log.info(f'{self.pl}: Saved Results for all symbols!')
            self.log.info(f'{self.pl}: Results of {len(self._params_list)} Models execution: \n{pd.DataFrame(results, columns=["status"])["status"].value_counts()}')
