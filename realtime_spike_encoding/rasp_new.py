from multiprocessing import Process, Queue, Pool
from kafka import KafkaProducer
import json
from ctypes import *
from dwfconstants import *
import time
from multiprocessing import Queue
import numpy as np


dwf = cdll.dwf
producer = KafkaProducer(bootstrap_servers=['172.17.39.222:9092'],
                         value_serializer=lambda v:
                         json.dumps(v).encode('utf-8'))


previous_states = {
    'lif': {'v': -65, 'tlast': 0},
    'lif2': {'v': -65, 'tlast': 0},
    'lif3': {'v': -65, 'tlast': 0},
    }


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


def lif(current, t, th=-45, rest=-65, lif_time=0.2, ref=3, tc_decay=100):
    """ simple LIF neuron """
    state = previous_states['lif']
    v = state['v']
    tlast = state['tlast']

    vpeak = 20  # 膜電位のピーク(最大値)
    dv = ((lif_time * t) > (tlast + ref)) * (-v + rest + current) / tc_decay  # 微小膜電位増加量
    v = v + lif_time * dv  # 膜電位を計算
    tlast = tlast + (lif_time * t - tlast) * (v >= th)  # 発火したら発火時刻を記録
    v = v + (vpeak - v) * (v >= th)  # 発火したら膜電位をピークへ
    spike = 1 if v >= th else 0  # 閾値を超えたら1、そうでなければ0を返す
    if v >= th:
        v = rest  # 閾値を超えたときのみ静止膜電位に戻す

    # 状態を保存
    previous_states['lif']['v'] = v
    previous_states['lif']['tlast'] = tlast

    return spike


def lif2(current, t, th=-58, rest=-65, lif_time=0.2, ref=3, tc_decay=100):
    """ simple LIF neuron """
    state = previous_states['lif2']
    v = state['v']
    tlast = state['tlast']

    vpeak = 20  # 膜電位のピーク(最大値)
    dv = ((lif_time * t) > (tlast + ref)) * (-v + rest + current) / tc_decay  # 微小膜電位増加量
    v = v + lif_time * dv  # 膜電位を計算
    tlast = tlast + (lif_time * t - tlast) * (v >= th)  # 発火したら発火時刻を記録
    v = v + (vpeak - v) * (v >= th)  # 発火したら膜電位をピークへ
    spike = 1 if v >= th else 0  # 閾値を超えたら1、そうでなければ0を返す
    if v >= th:
        v = rest  # 閾値を超えたときのみ静止膜電位に戻す

    # 状態を保存
    previous_states['lif2']['v'] = v
    previous_states['lif2']['tlast'] = tlast

    return spike


def lif3(current, t, th=-40, rest=-65, lif_time=0.2, ref=3, tc_decay=100):
    """ simple LIF neuron """
    state = previous_states['lif3']
    v = state['v']
    tlast = state['tlast']
    vpeak = 20  # 膜電位のピーク(最大値)
    max_current = 1000  # 最大電流の調整パラメータ
    currents_new = max_current / (np.abs(current) + 0.000000001) # 0除算を防ぐための小さい値
    dv = ((lif_time * t) > (tlast + ref)) * (-v + rest + currents_new) / tc_decay  # 微小膜電位増加量
    v = v + lif_time * dv  # 膜電位を計算
    tlast = tlast + (lif_time * t - tlast) * (v >= th)  # 発火したら発火時刻を記録
    v = v + (vpeak - v) * (v >= th)  # 発火したら膜電位をピークへ
    spike = 1 if v >= th else 0  # 閾値を超えたら1、そうでなければ0を返す
    if v >= th:
        v = rest  # 閾値を超えたときのみ静止膜電位に戻す

    # 状態を保存
    previous_states['lif3']['v'] = v
    previous_states['lif3']['tlast'] = tlast

    return spike


def poisson1(current, lif_time=0.2):
    """ Poisson encoding """
    # スパイクレートの計算（正規化なし）
    spike_rate = np.abs(current)  # スパイクレートを直接振幅に依存させる
    if np.random.poisson(spike_rate * lif_time / 1000) > 1:
        spike = 1
    else:
        spike = 0
        return spike
    

def poisson2(current, lif_time=0.2):
    """ Poisson encoding """
    # スパイクレートの計算（正規化なし）
    spike_rate = np.abs(current)
    max_spike_rate = 100
    spike_rates = max_spike_rate / (spike_rate + 0.00000001)
    if np.random.poisson(spike_rates * lif_time / 1000) > 0.5:
        spike = 1
    else:
        spike = 0
        return spike


def poisson3(current, lif_time=0.2):
    """ Poisson encoding """
    # スパイクレートの計算（正規化なし）
    spike_rate = np.abs(current)
    max_spike_rate = 100000
    spike_rates = max_spike_rate / (spike_rate + 0.00000001)
    if np.random.poisson(spike_rates * lif_time / 100000) > 1:
        spike = 1
    else:
        spike = 0
        return spike


def spike_encoding(queue):
    t = 0
    while True:
        if not queue.empty():
            current = queue.get()  # キューから現在の電流値を取得

            # 5つのエンコーディング関数を並列に実行
            with Pool(6) as pool:
                results = pool.starmap(
                    lambda func, args: func(*args),
                    [
                        (lif, (current, t)),
                        (lif2, (current, t)),
                        (lif3, (current, t)),
                        (poisson1, (current)),
                        (poisson2, (current)),
                        (poisson3, (current))
                    ]
                )

            # 結果をタプルで取得
            spike_tuple = (
                results[0][-1],  # 最後のスパイク結果
                results[1][-1],
                results[2][-1],
                results[3][-1],
                results[4][-1]
            )

            # Kafkaに結果を送信
            producer.send('spike-results', {'time': t, 'spikes': spike_tuple})

            t += 1  # 時間ステップをインクリメント
        else:
            # キューが空の場合は少し待機して再チェック
            time.sleep(0.01)


if __name__ == '__main__':
    queue = Queue()
    run_time = 10  # 10秒間シミュレーションを行う

    # プロセスの作成
    producer1 = Process(target=collect_data, args=(queue, run_time))
    consumer1 = Process(target=spike_encoding, args=(queue))

    # プロセスの開始
    producer1.start()
    consumer1.start()

    # プロセスの終了待ち
    producer1.join()
    consumer1.join()
