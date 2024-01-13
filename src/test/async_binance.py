from binance import AsyncClient
import time
import asyncio

import sys
sys.path.insert(0, sys.path[0].removesuffix('/src/test'))
import src.utils_binance as utils
import src.utils as utils2
import src.myenv as myenv

myenv.telegram_key.append(utils2.get_telegram_key())
loop = asyncio.new_event_loop()
api_key = '<api_key>'
api_secret = '<api_secret>'


def main():
    api_key, api_secret = utils.get_keys()
    loop = asyncio.get_event_loop()

    # client = loop.run_until_complete(AsyncClient.create(api_key, api_secret))
    client = loop.run_until_complete(AsyncClient.create(api_key, api_secret, requests_params={'timeout': 20}, loop=loop))
    print(f'Client>: {client}')

    res = loop.run_until_complete(client.get_account())
    # res = client.get_account()
    print(f'get_account>: {res}')

    loop.run_until_complete(client.close_connection())


if __name__ == "__main__":
    main()
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(main())
