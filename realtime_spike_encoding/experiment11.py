from multiprocessing import Process, Queue
from concurrent.futures import ThreadPoolExecutor
from ctypes import *
from dwfconstants import *
import math
import time
import sys
import numpy as np
import json

def initialize_device():
    if sys.platform.startswith("win"):
        dwf = cdll.dwf
    elif sys.platform.startswith("darwin"):
        dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
    else:
        dwf = cdll.LoadLibrary("libdwf.so")
    hdwf = c_int()
    hzAcq = c_double(5000)

    #print(DWF version
    version = create_string_buffer(16)
    dwf.FDwfGetVersion(version)
    print("DWF Version: "+str(version.value))

    dwf.FDwfParamSet(DwfParamOnClose, c_int(0))
    #open device
    print("Opening first device")
    dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

    if hdwf.value == hdwfNone.value:
        szerr = create_string_buffer(512)
        dwf.FDwfGetLastErrorMsg(szerr)
        print(str(szerr.value))
        print("failed to open device")
        quit()

    dwf.FDwfDeviceAutoConfigureSet(hdwf, c_int(0)) # 0 = the device will only be configured when FDwf###Configure is called

    #set up acquisition
    dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(0), c_int(1))
    dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(0), c_double(5))
    dwf.FDwfAnalogInAcquisitionModeSet(hdwf, acqmodeRecord)
    dwf.FDwfAnalogInFrequencySet(hdwf, hzAcq)
    dwf.FDwfAnalogInRecordLengthSet(hdwf, c_double(60))
    dwf.FDwfAnalogInChannelFilterSet(hdwf, c_int(-1), filterDecimate)
    dwf.FDwfAnalogInConfigure(hdwf, c_int(1), c_int(0))

    #wait at least 2 seconds for the offset to stabilize
    time.sleep(2)
    return hdwf, dwf




previous_states = {
    'lif': {'v': -65, 'tlast': 0},
    'lif2': {'v': -65, 'tlast': 0},
    'lif3': {'v': -65, 'tlast': 0},
    }


def collect_data(queue):
    hdwf, dwf = initialize_device()

    cAvailable = c_int()
    cLost = c_int()
    cCorrupted = c_int()
    sts = c_byte()
    fLost = 0
    fCorrupted = 0
    cSamples = 0
    nSamples = 10000
    print("Starting oscilloscope")
    dwf.FDwfAnalogInConfigure(hdwf, c_int(0), c_int(1))
    trigger_level = 0.15  # 閾値を1.0Vに設定
    trigger_started = False


    while cSamples < nSamples:
        dwf.FDwfAnalogInStatus(hdwf, c_int(1), byref(sts))
        if cSamples == 0 and (sts == DwfStateConfig or sts == DwfStatePrefill or sts == DwfStateArmed) :
            # Acquisition not yet started.
            continue

        dwf.FDwfAnalogInStatusRecord(hdwf, byref(cAvailable), byref(cLost), byref(cCorrupted))
        
        cSamples += cLost.value

        if cLost.value :
            fLost = 1
        if cCorrupted.value :
            fCorrupted = 1

        if cAvailable.value==0 :
            continue
        if not trigger_started:
            tmp_samples = (c_double * cAvailable.value)()
            dwf.FDwfAnalogInStatusData(hdwf, c_int(0), byref(tmp_samples), cAvailable)
            list(tmp_samples[:cAvailable.value])
            for sample in tmp_samples:
                if sample >= trigger_level:
                    trigger_started = True
                    print("starting recording...")
                    break

        if trigger_started:
            if cSamples + cAvailable.value > nSamples:
                cAvailable = c_int(nSamples - cSamples)
            rgdSamples = (c_double * cAvailable.value)()
            dwf.FDwfAnalogInStatusData(hdwf, c_int(0), rgdSamples, cAvailable) # get channel 1 data　おそらくunreaddataから読まれていく仕組み
            queue.put(list(rgdSamples[:cAvailable.value]))
            cSamples += cAvailable.value
    print("終わった")
    queue.put(None)
    dwf.FDwfAnalogOutReset(hdwf, c_int(0))
    dwf.FDwfDeviceCloseAll()


def lif(item, t, th=-45, rest=-65, lif_time=0.2, ref=3, tc_decay=100):
    """ simple LIF neuron """
    state = previous_states['lif']
    v = state['v']
    tlast = state['tlast']
    spikes = []
    for current in item:
        if (t * lif_time) > (tlast + ref):
            dv = (-v + rest + 100*current) / tc_decay
            v += lif_time * dv

            if v >= th:
                tlast = t * lif_time
                spike = 1   # producer.send('rasp-start', {'time': t})
                v = rest
            else:
                spike = 0
        else:
            spike = 0
        spikes.append(spike)
        previous_states['lif']['v'] = v
        previous_states['lif']['tlast'] = tlast
    return spikes


def lif2(item, t, th=-58, rest=-65, lif_time=0.2, ref=3, tc_decay=100):
    """ simple LIF neuron """
    state = previous_states['lif2']
    v = state['v']
    tlast = state['tlast']
    spikes = []
    for current in item:
        if (t * lif_time) > (tlast + ref):
            dv = (-v + rest + 100*current) / tc_decay
            v += lif_time * dv

            if v >= th:
                tlast = t * lif_time
                spike = 1   # producer.send('rasp-start', {'time': t})
                v = rest
            else:
                spike = 0
        else:
            spike = 0
        spikes.append(spike)
        previous_states['lif2']['v'] = v
        previous_states['lif2']['tlast'] = tlast
    return spikes


def lif3(item, t, th=-40, rest=-65, lif_time=0.2, ref=3, tc_decay=100):
    """ simple LIF neuron """
    state = previous_states['lif3']
    v = state['v']
    tlast = state['tlast']
    spikes = []
    max_current = 1000
    for current in item:
        currents_new = max_current / (np.abs(100*current) + 0.000000001)
        if (t * lif_time) > (tlast + ref):
            dv = (-v + rest + currents_new) / tc_decay
            v += lif_time * dv

            if v >= th:
                tlast = t * lif_time
                spike = 1   # producer.send('rasp-start', {'time': t})
                v = rest
            else:
                spike = 0
        else:
            spike = 0
        spikes.append(spike)
        previous_states['lif3']['v'] = v
        previous_states['lif3']['tlast'] = tlast
    return spikes


def poisson1(item, lif_time=0.2):
    """ Poisson encoding """
    # スパイクレートの計算
    spikes = []
    for current in item:
        spike_rate = np.abs(100*current)  # スパイクレートを直接振幅に依存させる
        if np.random.poisson(spike_rate * lif_time / 1000) > 1:
            spike = 1
        else:
            spike = 0
        spikes.append(spike)
    return spikes


def poisson2(item, lif_time=0.2):
    """ Poisson encoding """
    spikes = []
    for current in item:
        spike_rate = np.abs(100*current)
        max_spike_rate = 100
        spike_rates = max_spike_rate / (spike_rate + 0.00000001)
        if np.random.poisson(spike_rates * lif_time / 1000) > 0.5:
            spike = 1
        else:
            spike = 0
        spikes.append(spike)
    return spikes


def poisson3(item, lif_time=0.2):
    """ Poisson encoding """
    spikes = []
    for current in item:
        spike_rate = np.abs(100*current)
        max_spike_rate = 100000
        spike_rates = max_spike_rate / (spike_rate + 0.00000001)
        if np.random.poisson(spike_rates * lif_time / 100000) > 1:
            spike = 1
        else:
            spike = 0
        spikes.append(spike)
    return spikes


def call_function_with_args(func, *args):
    return func(*args)


def spike_encoding(queue):
    t = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        while True:
            item = queue.get()
            if item is None:
                break

            # 非同期で各関数を実行
            futures = [
                executor.submit(call_function_with_args, lif, item, t),
                executor.submit(call_function_with_args, lif3, item, t),
                executor.submit(call_function_with_args, poisson1, item),
                executor.submit(call_function_with_args, poisson2, item),
                executor.submit(call_function_with_args, lif2, item, t),
                executor.submit(call_function_with_args, poisson3, item)
            ]

            # 結果を集める
            results = [future.result() for future in futures]

            spike_tuple = (
                results[0],  
                results[1],
                results[2],
                results[3],
                results[4],
                results[5]
            )

            # Kafkaに結果を送信
            # print({'time': t, 'spikes': spike_tuple})

            t += 1



if __name__ == '__main__':

    queue = Queue()

    # プロセスの作成
    producer1 = Process(target=collect_data, args=(queue,))
    consumer1 = Process(target=spike_encoding, args=(queue,))

    # プロセスの開始
    producer1.start()
    consumer1.start()

    # プロセスの終了待ち
    producer1.join()
    consumer1.join()
