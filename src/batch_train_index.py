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


class BatchTrainIndex:
  def __init__(self,
               update_data_from_web,
               calc_rsi,
               verbose,
               log_level,
               symbol_list,
               interval_list,
               target_margin_list,
               range_min_rsi,
               range_max_rsi,
               range_p_ema,
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
    self._range_min_rsi = range_min_rsi
    self._range_max_rsi = range_max_rsi
    self._range_p_ema = range_p_ema

    self._range_p_ema_ini = int(range_p_ema[0])
    self._range_p_ema_end = int(range_p_ema[1])

    # print(f'***** BatchTrainIndex: \n{self._symbol_list}\n - {self._update_data_from_web}\n - {self._interval_list}\n - {self._target_margin_list}\n - {self._range_min_rsi}\n - {self._range_max_rsi}\n - {self._range_p_ema} *****')
    # sys.exit(0)

    # Private arguments
    self._all_data_list = {}

    # Initialize logging
    self.log = logging.getLogger("training_logger")

    # Prefix for log
    self.pl = f'BatchTrainIndex: '

    self._lock = None
    self._df_result_simulation = {}

  def _finalize_training(self, train_param):
    result = 'SUCESS'
    try:
      pnl = utils.simule_index_trading(
        train_param['all_data'],
        train_param['symbol'],
        train_param['interval'],
        train_param['p_ema'],
        myenv.default_amount_invested,
        train_param['target_margin'],
        train_param['range_min_rsi'],
        train_param['range_max_rsi'],
        train_param['stop_loss_multiplier'])

      # Lock Thread to register results
      ix_symbol = f'{train_param["symbol"]}_{train_param["interval"]}'
      self._lock.acquire()
      self._df_result_simulation[ix_symbol] = utils.save_index_results(
          self._df_result_simulation[ix_symbol],
          train_param['symbol'],
          train_param['interval'],
          train_param['target_margin'],
          train_param['p_ema'],
          train_param['range_min_rsi'],
          train_param['range_max_rsi'],
          myenv.default_amount_invested,
          pnl,
          train_param['stop_loss_multiplier'],
          train_param['arguments'],
          False)
      self._lock.release()
    except Exception as e:
      self.log.exception(e)
      result = 'ERROR'
    return result

  def _data_collection(self):
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

          _aux_data = utils.truncate_data_in_days(_aux_data, myenv.days_to_validate_train)
          _aux_data.info() if self._verbose else None
          self.log.info(f'{self.pl}: Store data in memory for symbol: {ix_symbol}...')
          if _aux_data.shape[0] == 0:
            raise Exception(f'Data for symbol: {ix_symbol} is empty')
          self._all_data_list[ix_symbol] = _aux_data
          self.log.info(f'{self.pl}: Loaded data to memory for symbol: {ix_symbol} - shape: {_aux_data.shape}')
        except Exception as e:
          self.log.exception(e)

  def _data_preprocessing(self):
    self.log.info('Start Data  Preprocessing...')
    for interval in self._interval_list:
      for symbol in self._symbol_list:
        try:
          ix_symbol = f'{symbol}_{interval}'
          self.log.info(f'{self.pl}: Calculating RSI for symbol: {ix_symbol}...')
          self._all_data_list[ix_symbol] = calc_utils.calc_RSI(self._all_data_list[ix_symbol])

          self.log.info(f'{self.pl}: Calculating EMA\'s for key {ix_symbol}...')
          self._all_data_list[ix_symbol] = calc_utils.calc_ema_periods(self._all_data_list[ix_symbol], periods_of_time=[i for i in range(self._range_p_ema_ini, self._range_p_ema_end + 1, 25)], diff_price=False)
          self._all_data_list[ix_symbol].dropna(inplace=True)

          self.log.info(f'{self.pl}:  info after calculating RSI and EMA\'s: ') if self._verbose else None
          self._all_data_list[ix_symbol].info() if self._verbose else None
        except Exception as e:
          self.log.exception(e)

  def run(self):
    self.log.info(f'{self.pl}: {self.__class__.__name__}: Start _data_collection...')
    self._data_collection()
    self.log.info(f'{self.pl}: {self.__class__.__name__}: Start _data_preprocessing...')
    self._data_preprocessing()
    self.log.info(f'{self.pl}: {self.__class__.__name__}: Start Running...')

    with Pool(processes=20) as pool:
      manager = Manager()
      df_result_simulation_list = manager.dict()
      lock = manager.Lock()

      params_list = []
      _prm_list = []
      for interval in self._interval_list:
        for symbol in self._symbol_list:
          ix_symbol = f'{symbol}_{interval}'
          df_result_simulation_list[ix_symbol] = utils.get_index_results(symbol, interval)
          for target_margin in self._target_margin_list:
            for p_ema in range(self._range_p_ema_ini, self._range_p_ema_end + 1, 25):
              for range_min_rsi in range(self._range_min_rsi, 36 + 1, 2):
                for range_max_rsi in range(72, self._range_max_rsi + 1, 2):
                  for stop_loss_multiplier in range(2, myenv.stop_loss_range_multiplier + 1):
                    train_param = {
                        'all_data': self._all_data_list[ix_symbol],
                        'symbol': symbol,
                        'interval': interval,
                        'target_margin': float(target_margin),
                        'range_min_rsi': int(range_min_rsi),
                        'range_max_rsi': int(range_max_rsi),
                        'p_ema': int(p_ema),
                        'stop_loss_multiplier': int(stop_loss_multiplier),
                        'calc_rsi': self._calc_rsi,
                        'verbose': self._verbose,
                        'lock': self._lock,
                        'arguments': str(sys.argv[1:])}
                    params_list.append(train_param)
                    _prm_list.append(train_param.copy())

      self.log.info(f'{self.pl}: Total Trainning Models: {len(params_list)}')
      for _prm in _prm_list:
        del _prm['all_data']
        del _prm['lock']
      pd.DataFrame(_prm_list).to_csv(f'{myenv.datadir}/params_list{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv', index=False)

      self.log.info(f'{self.pl}: Will Start {len(params_list)} Threads.')
      params_list = sorted(params_list, key=lambda p: (str(p['p_ema']), str(p['target_margin']), str(p['range_max_rsi']), str(p['range_min_rsi']), str(p['symbol']), str(p['interval'])))
      process_list = []
      count = 0
      for p in params_list:
        ix_symbol = f"{p['symbol']}_{p['interval']}"
        _key_tm = f"{ix_symbol}_{p['target_margin']}"
        name = f"{_key_tm}_{p['p_ema']}_{p['range_min_rsi']}_{p['range_max_rsi']}_{p['stop_loss_multiplier']}"
        try:
          if not utils.has_index_results(df_result_simulation_list[ix_symbol], p['symbol'], p['interval'], p['target_margin'], p['p_ema'], p['range_min_rsi'], p['range_max_rsi'], p['stop_loss_multiplier']):
            process = pool.apply_async(func=utils._finalize_index_train, kwds={'train_param': p, 'lock': lock, 'df_result_simulation_list': df_result_simulation_list, 'count': count})
            process_list.append({'name': name, 'symbol': p['symbol'], 'interval': p['interval'], 'target_margin': p['target_margin'], 'process': process})
            count += 1
        except Exception as e:
          self.log.exception(e)

      self.log.info(f'{self.pl}: Will Start collecting results for {len(params_list)} Threads.')
      results = []
      for p in process_list:
        try:
          res = p['process'].get()
          results.append(res)
        except Exception as e:
          self.log.exception(e)
          results.append("TIMEOUT_ERROR")

      self.log.info(f'{self.pl}: Saving Results for all symbols...')
      utils.save_all_results_index_simulation(df_result_simulation_list)
      self.log.info(f'{self.pl}: Saved Results for all symbols!')
      self.log.info(f'{self.pl}: Results of {len(params_list)} Models execution: \n{pd.DataFrame(results, columns=["status"])["status"].value_counts()}')
