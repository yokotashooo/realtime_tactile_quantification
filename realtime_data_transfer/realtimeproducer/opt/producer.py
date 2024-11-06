import numpy as np
from kafka import KafkaProducer
import json
import time

# データの読み込み
test_data_path = '/sampleexperiment.txt'
data = np.loadtxt(test_data_path, dtype='float')[::2]

# KafkaProducerの設定
producer = KafkaProducer(
    bootstrap_servers=['broker:29092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)


for i in range(len(data)):
    # start_time = time.perf_counter()
    record = {'index': i, 'values': data[i].tolist(), 'timestamp': time.time()}
    print(record)
    producer.send('experiment', value=record)  # keyを指定せずに送信
    # producer.flush()
    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time
    # sleep_time = max(0, interval - elapsed_time) 
    # time.sleep(sleep_time)
print("All data sent.")
