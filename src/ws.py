from binance.client import Client
from binance.streams import BinanceSocketManager

client = Client()

bm = BinanceSocketManager(client)
def process_m_message(msg):
	print("stream: {} data: {}".format(msg['stream'], msg['data']))

conn_key = bm.start_multiplex_socket(['sxpusdt@aggTrade', 'sxpusdt@ticker','sxpusdt@markPrice'], process_m_message)
bm.start()