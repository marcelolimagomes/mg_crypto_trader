from binance import ThreadedWebsocketManager
from multiprocessing import Pool, Process

from src.robo_trader_ml import RoboTraderML
# from src.robo_trader_index import RoboTraderIndex
from src.robo_trader_ix_binance import RoboTraderIndex

import src.utils as utils
import src.calcEMA as calc_utils
import src.myenv as myenv

import logging
import threading
import os


class BatchRoboTrader:
    def __init__(self,
                 verbose,
                 start_date,
                 prediction_mode,
                 update_data_from_web,
                 log_level):

        # Boolean arguments
        self._verbose = verbose
        self._update_data_from_web = update_data_from_web
        # Single arguments
        self._start_date = start_date
        self._prediction_mode = prediction_mode
        self._log_level = log_level
        # Private arguments
        self._all_data_list = {}
        self._top_params = None
        if self._prediction_mode == "ml":
            self._top_params = utils.get_best_parameters_ml()

        self._top_params_index = None
        if self._prediction_mode == "index":
            self._top_params_index = utils.get_best_parameters_index()

        # Initialize logging
        self.log = self._configure_log(log_level)
        self.log.setLevel(log_level)

    def _configure_log(self, log_level):
        log_file_path = os.path.join(myenv.logdir, myenv.batch_robo_log_filename)
        logger = logging.getLogger("batch_robo_logger")
        logger.propagate = False
        logger.setLevel(log_level)

        fh = logging.FileHandler(log_file_path, mode='a', delay=True)
        fh.setFormatter(logging.Formatter(f'[%(asctime)s] - %(levelname)s - {self.__class__.__name__}: %(message)s', '%Y-%m-%d %H:%M:%S'))
        fh.setLevel(log_level)

        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(f'[%(asctime)s] - %(levelname)s - {self.__class__.__name__}: %(message)s', '%Y-%m-%d %H:%M:%S'))
        sh.setLevel(log_level)

        logger.addHandler(fh)
        logger.addHandler(sh)
        return logger

    def _data_collection(self):
        if self._prediction_mode == "ml":
            for param in self._top_params:
                try:
                    ix_symbol = f'{param["symbol"]}_{param["interval"]}'
                    if ix_symbol not in self._all_data_list:
                        self.log.info(f'PredictionMode: {self._prediction_mode} - Loading data to memory for symbol: {ix_symbol}...')
                        self._all_data_list[ix_symbol] = utils.get_data(
                            symbol=f'{param["symbol"]}',
                            save_database=False,
                            interval=param['interval'],
                            tail=-1,
                            columns=myenv.all_cols,
                            parse_dates=True,
                            updata_data_from_web=self._update_data_from_web,
                            start_date=self._start_date)
                    self._all_data_list[ix_symbol] = self._all_data_list[ix_symbol].tail(1000)
                    self.log.info(f'Loaded data to memory for symbol: {ix_symbol}')
                except Exception as e:
                    self.log.error(e)
        elif self._prediction_mode == "index":
            for params in self._top_params_index:
                try:
                    ix_symbol = f"{params['symbol']}_{params['interval']}"
                    if ix_symbol not in self._all_data_list:
                        limit = int(params["p_ema"]) * 2
                        self.log.info(f'PredictionMode: {self._prediction_mode} - Loading data to memory for symbol: {ix_symbol} - Limit: {limit}...')
                        self._all_data_list[ix_symbol] = utils.get_klines(
                            symbol=params['symbol'],
                            interval=params['interval'],
                            max_date=None,
                            limit=limit,
                            columns=myenv.all_index_cols,
                            parse_dates=True)
                    self._all_data_list[ix_symbol].info() if self._verbose else None
                    self.log.info(f'Loaded data to memory for symbol: {ix_symbol} - Shape: {self._all_data_list[ix_symbol].shape}')
                except Exception as e:
                    self.log.error(e)

    def _data_preprocessing(self):
        self.log.info('Prepare All Data...')
        if self._prediction_mode == "ml":
            for param in self._top_params:
                ix_symbol = f'{param["symbol"]}_{param["interval"]}'
                try:
                    self.log.info(f'PredictionMode: {self._prediction_mode} - Calculating EMAs for key {ix_symbol}...')
                    self._all_data_list[ix_symbol] = calc_utils.calc_ema_periods(self._all_data_list[ix_symbol], periods_of_time=[int(param['times_regression_profit_and_loss']), 200])
                    self.log.info(f'info after calculating EMA\'s: ') if self._verbose else None
                    self._all_data_list[ix_symbol].info() if self._verbose else None

                    self.log.info(f'PredictionMode: {self._prediction_mode} - Calc RSI for symbol: {ix_symbol}')
                    self._all_data_list[ix_symbol] = calc_utils.calc_RSI(self._all_data_list[ix_symbol])
                    self._all_data_list[ix_symbol].dropna(inplace=True)
                    self._all_data_list[ix_symbol].info() if self._verbose else None

                except Exception as e:
                    self.log.error(e)
        elif self._prediction_mode == "index":
            for param in self._top_params_index:
                try:
                    ix_symbol = f'{param["symbol"]}_{param["interval"]}'
                    p_ema = int(param["p_ema"])
                    self.log.info(f'PredictionMode: {self._prediction_mode} - Calculating EMA\'s for key {ix_symbol}...')
                    self._all_data_list[ix_symbol] = calc_utils.calc_ema_periods(self._all_data_list[ix_symbol], periods_of_time=[p_ema], diff_price=False)
                    self.log.info(f'info after calculating EMAs: ') if self._verbose else None
                    self._all_data_list[ix_symbol].info() if self._verbose else None

                    self.log.info(f'PredictionMode: {self._prediction_mode} - Calc RSI for symbol: {ix_symbol}')
                    self._all_data_list[ix_symbol] = calc_utils.calc_RSI(self._all_data_list[ix_symbol])
                    # self._all_data_list[ix_symbol].dropna(inplace=True)
                    self._all_data_list[ix_symbol].info() if self._verbose else None

                except Exception as e:
                    self.log.error(e)
    # Public methods

    def run_robo_trader_ml(self):
        self.log.info(f'PredictionMode: {self._prediction_mode} - Start Running...')

        robo_trader_params_list = {}
        for params in self._top_params:
            model_name = utils.get_model_name_to_load(params['strategy'], params['symbol'], params['interval'], params['estimator'], params['stop_loss'],
                                                      params['regression_times'], params['times_regression_profit_and_loss'])
            if model_name is None:
                continue

            ix_symbol = f'{params["symbol"]}_{params["interval"]}'

            params['all_data'] = self._all_data_list[ix_symbol]
            params['start_date'] = self._start_date
            params['calc_rsi'] = '-calc-rsi' in params['arguments']
            params['verbose'] = self._verbose
            params['arguments'] = params['arguments']
            params['log_level'] = self._log_level

            strategy = params['strategy']
            if ix_symbol not in robo_trader_params_list:
                robo_trader_params_list[ix_symbol] = {}
                robo_trader_params_list[ix_symbol]['symbol'] = params['symbol']
                robo_trader_params_list[ix_symbol]['interval'] = params['interval']
                # robo_trader_params_list[ix_symbol]['mutex'] = mutex

            robo_trader_params_list[ix_symbol][strategy] = params
            '''
        model_name = utils.get_model_name_to_load(params['strategy'], params['symbol'], params['interval'], params['estimator'], params['stop_loss'],
            params['regression_times'], params['times_regression_profit_and_loss'])
        if model_name is None:
          raise Exception(f'Best model not found: {model_name}')
        '''

        self.log.info(f'Total Robo Trades to start...: {len(robo_trader_params_list)}')
        thread_list = []
        for ix_symbol in robo_trader_params_list:
            self.log.info(f'Starting Robo Trader for Symbol: {ix_symbol}')
            robo = RoboTraderML(robo_trader_params_list[ix_symbol])
            # thread = threading.Thread(target=robo.run, name=ix_symbol)
            # thread.start()
            # thread_list.append(thread)

    def run_robo_trader_index(self):
        self.log.info(f'PredictionMode: {self._prediction_mode} - Start Running...')

        with Pool(processes=os.cpu_count()) as pool:
            self.log.info(f'Total Robo Trades to start...: {len(self._top_params_index)}')
            for param in self._top_params_index:
                # print('twm>>>>> ', twm.start_kline_socket(callback=self.handle_socket_kline, args=param, symbol=param["symbol"], interval=param["interval"]))
                ix_symbol = f'{param["symbol"]}_{param["interval"]}'
                # param['all_data'] = self._all_data_list[ix_symbol]
                param['start_date'] = self._start_date
                # param['latest_update'] = utils.get_latest_update(self._all_data_list[ix_symbol])
                param['calc_rsi'] = True
                param['verbose'] = self._verbose
                # params['arguments'] = params['arguments']
                param['log_level'] = self._log_level
                # param['mutex'] = mutex
                # param['twm'] = twm
                self.log.info(f'Starting Robo Trader for Symbol: {ix_symbol}')
                robo = RoboTraderIndex(param)
                # self.log.info(f"Start ThreadedWebsocketManager: {twm.start_kline_socket(callback=robo.handle_socket_kline, symbol=param['symbol'], interval=param['interval'])}")
                process = Process(target=robo.run, name=ix_symbol)
                process.start()
                # thread = threading.Thread(target=robo.run, name=ix_symbol)
                # thread.start()

        # self.log.info(f'Calling ThreadedWebsocketManager.join()')
        # twm.join()

    def run(self):

        if self._prediction_mode == "ml":
            self.log.info(f'Start _data_collection...')
            self._data_collection()
            self.log.info(f'Start _data_preprocessing...')
            self._data_preprocessing()
            self.run_robo_trader_ml()
        elif self._prediction_mode == "index":
            self.run_robo_trader_index()
