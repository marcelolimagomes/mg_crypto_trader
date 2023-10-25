from src.trainig import Train

import src.utils as utils
import src.calcEMA as calc_utils
import src.myenv as myenv
import logging
import pandas as pd


class TrainBestModel:
  def __init__(self,
               verbose,
               log_level):

    # Boolean arguments
    self._verbose = verbose
    # Single arguments
    self._log_level = log_level
    # Private arguments
    self._all_data_list = {}
    self._top_params = utils.get_best_parameters()
    # Initialize logging
    self.log = logging.getLogger("training_logger")

  def _data_collection(self):
    self.log.info(f'Loading data to memory: Symbols: {[s["symbol"] for s in self._top_params]} - Intervals: {[s["interval"] for s in self._top_params]}')
    for param in self._top_params:
      try:
        ix_symbol = utils.get_ix_symbol(param["symbol"], param["interval"], param["stop_loss"], param["times_regression_profit_and_loss"])
        if ix_symbol not in self._all_data_list:
          self.log.info(f'Loading data for symbol: {ix_symbol}...')
          _aux_data = utils.get_data(
              symbol=param["symbol"],
              save_database=False,
              interval=param['interval'],
              tail=-1,
              columns=myenv.all_cols,
              parse_dates=True,
              updata_data_from_web=False)
          
          _aux_data = _aux_data.tail(myenv.rows_to_train)
          self._all_data_list[ix_symbol] = _aux_data
          self.log.info(f'Loaded data for symbol: {ix_symbol} - shape: {_aux_data.shape}')
          _aux_data.info() if self._verbose else None
      except Exception as e:
        self.log.error(e)
    self.log.info(f'Loaded data to memory for symbols: {[s["symbol"] for s in self._top_params]}')

  def _data_preprocessing(self):
    self.log.info('Prepare All Data...')
    for param in self._top_params:
      ix_symbol = utils.get_ix_symbol(param["symbol"], param["interval"], param["stop_loss"], param["times_regression_profit_and_loss"])
      try:
        self.log.info(f'Calculating EMA\'s for key {ix_symbol}...')
        if 'ema_200p' not in self._all_data_list[ix_symbol].columns:
          self._all_data_list[ix_symbol] = calc_utils.calc_ema_periods(self._all_data_list[ix_symbol], periods_of_time=[int(param['times_regression_profit_and_loss']), 200])
          self.log.info(f'info after calculating EMA\'s: ') if self._verbose else None
          self._all_data_list[ix_symbol].info() if self._verbose else None

        if 'rsi' not in self._all_data_list[ix_symbol].columns:
          self.log.info(f'Calc RSI for symbol: {ix_symbol}')
          self._all_data_list[ix_symbol] = calc_utils.calc_RSI(self._all_data_list[ix_symbol])
          #self._all_data_list[ix_symbol].dropna(inplace=True)
          self.log.info('info after CalcRSI start_date: ') if self._verbose else None
          self._all_data_list[ix_symbol].info() if self._verbose else None

        if myenv.label not in self._all_data_list[ix_symbol]:
          self.log.info(f'calculating regression_profit_and_loss - times: {int(param["times_regression_profit_and_loss"])} - stop_loss: {float(param["stop_loss"])}')
          self._all_data_list[ix_symbol] = utils.regression_PnL(
              data=self._all_data_list[ix_symbol],
              label=myenv.label,
              diff_percent=float(param['stop_loss']),
              max_regression_profit_and_loss=int(param['times_regression_profit_and_loss']),
              drop_na=True,
              drop_calc_cols=True,
              strategy=None)
          self.log.info('info after calculating regression_profit_and_loss: ')
          self._all_data_list[ix_symbol].info() if self._verbose else None

      except Exception as e:
        self.log.error(e)
  
  def run(self):
    self.log.info(f'{self.__class__.__name__}: Start _data_collection...')
    self._data_collection()
    self.log.info(f'{self.__class__.__name__}: Start _data_preprocessing...')
    self._data_preprocessing()

    params_list = []
    for param in self._top_params:
      n_jobs = -1
      fold = 3
      _param = param['arguments'].replace('[', '').replace(']', '').replace('\'', '').replace(' ', '').split(',')
      for p in _param:
        if (p.startswith('-n-jobs=')):
          n_jobs = int(p.split('=')[1])

        if (p.startswith('-fold=')):
          fold = int(p.split('=')[1])

      ix_symbol = utils.get_ix_symbol(param["symbol"], param["interval"], param["stop_loss"], param["times_regression_profit_and_loss"])
      train_param = {
          'all_data': self._all_data_list[ix_symbol],
          'strategy': param['strategy'],
          'symbol': param['symbol'],
          'interval': param['interval'],
          'estimator': param['estimator'],
          'imbalance_method': param['imbalance_method'],
          'train_size': myenv.train_size,
          'start_train_date': param['start_train_date'],
          'start_test_date': None,
          'numeric_features': param['numeric_features'],
          'stop_loss': param['stop_loss'],
          'regression_times': param['regression_times'],
          'regression_features': param['regression_features'],
          'times_regression_profit_and_loss': param['times_regression_profit_and_loss'],
          'calc_rsi': '-calc-rsi' in param['arguments'],
          'compare_models': False,
          'n_jobs': n_jobs,
          'use_gpu': '-use-gpu' in param['arguments'],
          'verbose': self._verbose,
          'normalize': '-normalize' in param['arguments'],
          'fold': fold,
          'use_all_data_to_train': True,
          'arguments': param['arguments'],
          'no_tune': '-no-tune' in param['arguments'],
          'save_model': True}
      params_list.append(train_param)

    results = []
    for params in params_list:
      train = Train(params)
      res = train.run()
      results.append(res)

    self.log.info(f'Results of {len(params_list)} Models execution: \n{pd.DataFrame(results, columns=["status"])["status"].value_counts()}')
