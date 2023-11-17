import src.utils as utils
import numpy as np
from pycaret.classification import ClassificationExperiment
from imblearn.under_sampling import InstanceHardnessThreshold, RepeatedEditedNearestNeighbours, AllKNN

import src.myenv as myenv
import logging
import sys


class TrainIndex:
  def __init__(self, params: dict):
    self.log = logging.getLogger("training_logger")

    # Single arguments
    self._all_data = params['all_data']
    self._symbol = params['symbol']
    self._interval = params['interval']
    self._target_margin = float(params['target_margin'])

    self._min_rsi = params['range_min_rsi']
    self._max_rsi = params['range_max_rsi']
    self._p_ema = params['p_ema']
    self._stop_loss_multiplier = params['stop_loss_multiplier']

    self._lock = params['lock']

    # Boolean arguments
    self._calc_rsi = params['calc_rsi']
    self._verbose = params['verbose']
    self._arguments = params['arguments']

    # Prefix for log
    self.pl = f'Train: {self._symbol}-{self._interval}-TM-{self._target_margin}'

  def _finalize_training(self):
    amount_invested = 100.0
    pnl = utils.simule_index_trading(
      self._all_data,
      self._symbol,
      self._interval,
      self._p_ema,
      amount_invested,
      self._target_margin,
      self._min_rsi,
      self._max_rsi,
      self._stop_loss_multiplier)

    # Lock Thread to register results
    self._lock.acquire()
    utils.save_index_results(
        self._symbol,
        self._interval,
        self._target_margin,
        self._p_ema,
        self._min_rsi,
        self._max_rsi,
        amount_invested,
        pnl,
        self._stop_loss_multiplier,
        self._arguments)
    self._lock.release()

  def run(self):
    result = 'SUCESS'
    try:
      self.log.info(f'{self.pl}: \n\nStart Trainign >>>>>>>>>>>>>>>>>>>>>>')
      self._finalize_training()
      self.log.info(f'{self.pl}: End Trainign <<<<<<<<<<<<<<<<<<<<<<\n\n')
    except Exception as e:
      self.log.exception(e)
      result = 'ERROR'
    return result
