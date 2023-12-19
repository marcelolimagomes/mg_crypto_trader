import src.utils as utils
import src.calcEMA as calc_utils
import src.myenv as myenv
import src.send_message as sm
import logging
import pandas as pd
import numpy as np
from multiprocessing import Process, Lock, Pool, Manager
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
        for interval in self._interval_list:
            for symbol in self._symbol_list:
                ix_symbol = f'{symbol}_{interval}'
                self._calc_indexes(ix_symbol)
                for p_ema in range(self._range_p_ema_ini, self._range_p_ema_end + 1, 25):
                    for min_rsi in range(self._min_rsi, 36 + 1, 2):
                        for max_rsi in range(72, self._max_rsi + 1, 2):
                            ix_predict = f'{symbol}_{interval}_{p_ema}_{min_rsi}_{max_rsi}'
                            _data = self._all_data_list[ix_symbol]
                            _data[ix_predict] = np.where((_data['close'] > _data[f'ema_{p_ema}p']) & (_data['rsi'] >= max_rsi), 'SHORT', np.where((_data['close'] < _data[f'ema_{p_ema}p']) & (_data['rsi'] <= min_rsi), 'LONG', 'ESTAVEL'))

    def run(self):
        self._data_collection()
        self._data_preprocessing()
        self.log.info(f'{self.pl}: Start Running...')

        # os.cpu_count()
        with Pool(processes=os.cpu_count()) as pool:
            self.log.info(f'{self.pl}: Total Trainning Models: {len(self._params_list)}')
            self.log.info(f'{self.pl}: Will Start {len(self._params_list)} Threads.')
            process_list = []
            count = 0
            for interval in self._interval_list:
                for symbol in self._symbol_list:
                    ix_symbol = f'{symbol}_{interval}'
                    p = {}
                    p['symbol'] = symbol
                    p['interval'] = interval
                    p['_data'] = self._all_data_list[ix_symbol]
                    p['range_p_ema_ini'] = int(self._range_p_ema_ini)
                    p['range_p_ema_end'] = int(self._range_p_ema_end)
                    p['range_min_rsi'] = int(self._min_rsi)
                    p['range_max_rsi'] = int(self._max_rsi)
                    p['target_margin_list'] = self._target_margin_list
                    p['start_amount_invested'] = 100.00

                    # _data: pd.DataFrame, range_p_ema_ini, range_p_ema_end, range_min_rsi, range_max_rsi, target_margin_list, start_amount_invested=100
                    try:
                        process = pool.apply_async(func=utils.simule_index_trading2, kwds=p)
                        process_list.append({'name': ix_symbol, 'symbol': symbol, 'interval': interval, 'process': process})
                        count += 1
                    except Exception as e:
                        self.log.exception(e)
                        traceback.print_stack()

            self.log.info(f'{self.pl}: Will Start collecting results for {len(self._params_list)} Threads.')
            for p in process_list:
                try:
                    res = p['process'].get()
                    # res = utils.parse_simule_index_trading2(res)
                    df_result = pd.DataFrame(data=res)
                    df_result.info()
                    utils.only_save_index_results(df_result, p['symbol'], p['interval'])
                except Exception as e:
                    self.log.exception(e)
                    traceback.print_stack()

            self.log.info(f'{self.pl}: Results of {len(self._params_list)} saved!')
