from multiprocessing import Process, Queue
from kafka import KafkaProducer
import json
from ctypes import *
from dwfconstants import *
import time
from multiprocessing import Queue

dwf = cdll.dwf
producer = KafkaProducer(bootstrap_servers=['172.17.39.222:9092'],
                         value_serializer=lambda v:
                         json.dumps(v).encode('utf-8'))


rest = -65
th = -52
ref = 3
tc_decay = 100
dt = 0.2


def collect_data(queue, run_time):
    # DWFデバイスの設定
    hdwf = c_int()
    dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))
    if hdwf.value == hdwfNone.value:
        print("failed to open device")
        return

    dwf.FDwfAnalogInFrequencySet(hdwf, c_double(5000.0))
    dwf.FDwfAnalogInBufferSizeSet(hdwf, c_int(10000))
    dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(0), c_int(1))
    dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(0), c_double(5))
    dwf.FDwfAnalogInChannelFilterSet(hdwf, c_int(0), filterDecimate)
    dwf.FDwfAnalogInConfigure(hdwf, c_int(1), c_int(1))

    # データ取得
    start_time = time.time()
    rgdSamples = (c_double*1000)()
    while time.time() - start_time < run_time:
        sts = c_byte()
        dwf.FDwfAnalogInStatus(hdwf, c_int(1), byref(sts))
        dwf.FDwfAnalogInStatusData(hdwf, 0, rgdSamples, 1000)
        # キューにデータを送信
        #for value in rgdSamples:
         #   queue.put(value)

    dwf.FDwfDeviceCloseAll()


def lif_algorithm(queue):
    v = rest
    tlast = 0
    t = 0
    while True:
        if not queue.empty():
            current = queue.get()  # キューから現在の電流値を取得
            dv = ((dt * t) > (tlast + ref)) * (-v + rest + current) / tc_decay
            v += dt * dv

            if v >= th:
                tlast = t * dt  # スパイク時の最終発火時間を更新
                # producer.send('rasp-start', {'time': t})
                v = rest  # 発火後、膜電位をリセット
            t += 1  # 時間ステップをインクリメント
        else:
            # キューが空の場合は少し待機して再チェック
            time.sleep(0.01)


if __name__ == '__main__':
    queue = Queue()
    run_time = 10  # 10秒間シミュレーションを行う

    # プロセスの作成
    producer1 = Process(target=collect_data, args=(queue, run_time))
    consumer1 = Process(target=lif_algorithm, args=(queue))

    # プロセスの開始
    producer1.start()
    consumer1.start()

    # プロセスの終了待ち
    producer1.join()
    consumer1.join()