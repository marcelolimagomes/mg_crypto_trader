import os
from pathlib import Path

file = Path(__file__)
parent = file.parent
os.chdir(parent)
print(file, parent, os.getcwd())


os.environ["PYCARET_CUSTOM_LOGGING_LEVEL"] = "CRITICAL"

import src.start_batch_training as bt
import src.start_robo_trader as bot
import src.myenv as myenv

import src.utils as utils
import sys
import logging


# Now you can use the 'argument' variable in your script
print("Argument provided:", sys.argv[1:])

log = None


def configure_log(log_level):
    log_file_path = os.path.join(myenv.logdir, myenv.main_log_filename)
    logger = logging.getLogger()
    logger.propagate = False

    logger.setLevel(log_level)
    fh = logging.FileHandler(log_file_path, mode='a', delay=True)
    fh.setFormatter(logging.Formatter('[%(asctime)s] - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S'))
    fh.setLevel(log_level)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(f'[%(asctime)s] - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S'))
    sh.setLevel(log_level)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


def main(args):
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    myenv.telegram_key.append(utils.get_telegram_key())

    log_level = logging.INFO
    interval_list = ['1h']
    start_date = '2010-01-01'
    auto_start_date = False
    tail = -1

    # Generic Params
    for arg in args:
        if (arg.startswith('-log-level=DEBUG')):
            log_level = logging.DEBUG
        if (arg.startswith('-log-level=WARNING')):
            log_level = logging.WARNING
        if (arg.startswith('-log-level=INFO')):
            log_level = logging.INFO
        if (arg.startswith('-log-level=ERROR')):
            log_level = logging.ERROR
        if (arg.startswith('-interval-list=')):
            aux = arg.split('=')[1]
            interval_list = aux.split(',')
        if (arg.startswith('-start-date=')):
            start_date = arg.split('=')[1]
        if (arg.startswith('-tail=')):
            tail = int(arg.split('=')[1])
        if '-auto-start-date' in args:
            auto_start_date = True

    log = configure_log(log_level)
    if '-download-data' in args:
        for interval in interval_list:
            if auto_start_date:
                _, aux_date = utils.get_start_date_for_interval(interval)
                start_date = aux_date.strftime("%Y-%m-%d")
            log.info(f'Starting download data, in interval ({interval}) auto-start-date: {auto_start_date} - start-date: {start_date} tail: {tail} for all Symbols in database...')
            utils.download_data(save_database=True, parse_dates=False, tail=tail, interval=interval, start_date=start_date)
        sys.exit(0)

    if '-prepare-best-parameters' in args:
        log.info('Starting prepare-best-parameters...')
        params = utils.prepare_best_params()
        log.info(params)
        sys.exit(0)

    if '-prepare-best-parameters-index' in args:
        log.info('Starting Prepare Best Parameters Index...')
        params = utils.prepare_best_params_index()
        log.info(params)
        sys.exit(0)

    if '-run-bot' in args:
        log.info('Starting bot...')
        bot.main(args)
        sys.exit(0)

    if '-batch-training' in args:
        log.info('Starting batch training...')
        bt.main(args)
        sys.exit(0)


if __name__ == '__main__':
    main(sys.argv[1:])
