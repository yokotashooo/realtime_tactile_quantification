from multiprocessing import Process, Queue
from ctypes import *
from dwfconstants import *
import math
import time
import sys
import numpy
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


def lif_algorithm(queue):
    rest = -65
    th = -64.88  #-52
    ref = 3
    tc_decay = 100
    dt = 0.2
    v = rest
    tlast = 0
    t = 0
    while True:
        item = queue.get()  # アイテムを取得
        if item is None:
            break  # Noneが受信されたらループを抜け、プロセスを終了       
        for current in item:
            if (t * dt) > (tlast + ref):
                dv = (-v + rest + current) / tc_decay
                v += dt * dv

            if v >= th:
                tlast = t * dt
                print(f"Spike at t = {t} ms")   # producer.send('rasp-start', {'time': t})
                v = rest  # 発火後、膜電位をリセット
            t += 1  # 時間ステップをインクリメント



def collect_data(queue):
    hdwf, dwf = initialize_device()

    cAvailable = c_int()
    cLost = c_int()
    cCorrupted = c_int()
    sts = c_byte()
    fLost = 0
    fCorrupted = 0
    cSamples = 0
    nSamples = 100000
    print("Starting oscilloscope")
    dwf.FDwfAnalogInConfigure(hdwf, c_int(0), c_int(1))
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

        if cSamples+cAvailable.value > nSamples :
            cAvailable = c_int(nSamples-cSamples)
        rgdSamples = (c_double * cAvailable.value)()
        dwf.FDwfAnalogInStatusData(hdwf, c_int(0), rgdSamples, cAvailable) # get channel 1 data　おそらくunreaddataから読まれていく仕組み
        queue.put(list(rgdSamples[:cAvailable.value]))

        cSamples += cAvailable.value
    print("終わった")
    queue.put(None)
    dwf.FDwfAnalogOutReset(hdwf, c_int(0))
    dwf.FDwfDeviceCloseAll()


if __name__ == '__main__':

    queue = Queue()

    # プロセスの作成
    producer1 = Process(target=collect_data, args=(queue,))
    consumer1 = Process(target=lif_algorithm, args=(queue,))

    # プロセスの開始
    producer1.start()
    consumer1.start()

    # プロセスの終了待ち
    producer1.join()
    consumer1.join()
