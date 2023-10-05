from src.robo_trader2 import RoboTrader
from src.trainig import Train

import src.utils as utils
import src.calcEMA as calc_utils
import src.myenv as myenv
import logging
import pandas as pd
import threading
import os
import time

mutex = threading.Lock()


class BatchRoboTrader:
  def __init__(self,
               verbose,
               start_date,
               log_level):

    # Boolean arguments
    self._verbose = verbose
    # Single arguments
    self._start_date = start_date
    self._log_level = log_level
    # Private arguments
    self._all_data_list = {}
    self._top_params = utils.get_best_parameters()

    # Initialize logging
    self.log = self._configure_log(log_level)
    self.log.setLevel(log_level)

  def _configure_log(self, log_level):
    log_file_path = os.path.join(myenv.logdir, myenv.batch_robo_log_filename)
    logger = logging.getLogger("batch_robo_logger")
    logger.propagate = False
    logger.setLevel(log_level)

    fh = logging.FileHandler(log_file_path, mode='a', delay=True)
    fh.setFormatter(logging.Formatter(f'[%(asctime)s] - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S'))
    fh.setLevel(log_level)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(f'[%(asctime)s] - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S'))
    sh.setLevel(log_level)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

  def _data_collection(self):
    self.log.info(f'Loading data to memory: Symbols: {[s["symbol"] for s in self._top_params]} - Intervals: {[s["interval"] for s in self._top_params]}')
    for param in self._top_params:
      try:
        ix_symbol = f'{param["symbol"]}_{param["interval"]}'
        self.log.info(f'Loading data for symbol: {ix_symbol}...')
        self._all_data_list[ix_symbol] = utils.get_data(
            symbol=f'{param["symbol"]}',
            save_database=False,
            interval=param['interval'],
            tail=-1,
            columns=myenv.all_cols,
            parse_dates=True,
            updata_data_from_web=False,
            start_date=self._start_date)
      except Exception as e:
        self.log.error(e)
    self.log.info(f'Loaded data to memory for symbols: {[s["symbol"] for s in self._top_params]}')

  def _data_preprocessing(self):
    self.log.info('Prepare All Data...')
    for param in self._top_params:
      ix_symbol = f'{param["symbol"]}_{param["interval"]}'
      try:
        self.log.info(f'Calc RSI for symbol: {ix_symbol}')
        self._all_data_list[ix_symbol] = calc_utils.calc_RSI(self._all_data_list[ix_symbol])
        self._all_data_list[ix_symbol].dropna(inplace=True)
        self._all_data_list[ix_symbol].info() if self._verbose else None

      except Exception as e:
        self.log.error(e)

  # Public methods

  def run(self):
    self.log.info(f'{self.__class__.__name__}: Start _data_collection...')
    self._data_collection()
    self.log.info(f'{self.__class__.__name__}: Start _data_preprocessing...')
    self._data_preprocessing()

    self.log.info(f'{self.__class__.__name__}: Start Running...')
    robo_trader_params_list = []
    for robo_trader_params in self._top_params:
      ix_symbol = f'{robo_trader_params["symbol"]}_{robo_trader_params["interval"]}'
      robo_trader_params = {
          'all_data': self._all_data_list[ix_symbol],
          'symbol': f'{robo_trader_params["symbol"]}',
          'interval': robo_trader_params['interval'],
          'estimator': robo_trader_params['estimator'],
          'start_date': self._start_date,
          'numeric_features': robo_trader_params['numeric_features'],
          'stop_loss': robo_trader_params['stop_loss'],
          'regression_times': robo_trader_params['regression_times'],
          'regression_features': robo_trader_params['regression_features'],
          'times_regression_profit_and_loss': robo_trader_params['times_regression_profit_and_loss'],
          'calc_rsi': '-calc-rsi' in robo_trader_params['arguments'],
          'verbose': self._verbose,
          'arguments': robo_trader_params['arguments'],
          'log_level': self._log_level,
          'mutex': mutex}
      robo_trader_params_list.append(robo_trader_params)

    self.log.info(f'Total Robo Trades to start...: {len(robo_trader_params_list)}')
    for robo_trader_params in robo_trader_params_list:
      model_name = utils.get_model_name_to_load(
          symbol=robo_trader_params['symbol'],
          interval=robo_trader_params['interval'],
          estimator=robo_trader_params['estimator'],
          stop_loss=robo_trader_params['stop_loss'],
          regression_times=robo_trader_params['regression_times'],
          times_regression_profit_and_loss=robo_trader_params['times_regression_profit_and_loss']
      )
      if model_name is None:
        raise Exception(f'Best model not found: {model_name}')

    thread_list = []
    for robo_trader_params in robo_trader_params_list:
      # print(params['symbol'], params['estimator'], params['stop_loss'], params['regression_times'], params['times_regression_profit_and_loss'])
      model_name = utils.get_model_name_to_load(
          symbol=robo_trader_params['symbol'],
          interval=robo_trader_params['interval'],
          estimator=robo_trader_params['estimator'],
          stop_loss=robo_trader_params['stop_loss'],
          regression_times=robo_trader_params['regression_times'],
          times_regression_profit_and_loss=robo_trader_params['times_regression_profit_and_loss']
      )

      thread_name = f'{robo_trader_params["symbol"]}_{robo_trader_params["interval"]}'
      self.log.info(f'Starting Robo Trader for Symbol: {thread_name}')
      robo = RoboTrader(robo_trader_params)
      thread = threading.Thread(target=robo.run, name=thread_name)
      thread.start()
      thread_list.append(thread)

    # Monitoring Thread Live
    self.log.info(f'Starting Thread Monitoring...')
    while True:
      for thread in thread_list:
        self.log.info(f'Thread name: {thread.name} - status: {thread.is_alive()}')
        if not thread.is_alive():
          thread_name = thread.name
          self.log.info(f'Removing dead thread name: {thread_name}')
          thread_list.remove(thread)
          for robo_trader_params in robo_trader_params_list:
            if f'{robo_trader_params["symbol"]}_{robo_trader_params["interval"]}' == thread_name:
              self.log.info(f'Starting new thread for symbol: {thread_name}')
              robo = RoboTrader(robo_trader_params)
              new_thread = threading.Thread(target=robo.run, name=f'params["symbol"]_{robo_trader_params["interval"]}')
              new_thread.start()
              thread_list.append(new_thread)
      # End Thread Validation
      time.sleep(60)
    # End While
