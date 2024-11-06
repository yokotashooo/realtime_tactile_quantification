# Realtime tactile quantification

* `realtime_data_transfer`  
Apache Kafkaを利用してデータをストリーミング転送するモジュールです．実験環境はこのプロジェクト内で設定されており，本番環境では研究室のLinuxサーバーを利用します．

  * `realtimeconsumer`  
    Apache Kafkaへのデータ送信を担当するモジュールです．

  * `realtimeproducer`  
    Apache Kafkaからデータを受信し，テスト分類処理を行うモジュールです．

* `realtime_spike_encoding`  
振動データをスパイク信号に変換するためのコード群です．このモジュールでは，センサデータをリアルタイムでスパイク化し，データ転送の準備を行います．

  * `data_collect.py`  
    振動が設定された閾値を超えた場合，2秒間のデータを収集します．
 
  * `realtime_encoding.py`  
    センサから取得した振動データをリアルタイムでスパイク信号に変換するWindows向けの実験用コードです.

  * `rasp.py`  
    Raspberry Pi上でセンサの振動をリアルタイムにスパイク化し，Apache Kafkaサーバーへ転送します（本番環境用）．

* `2classification.ipynb`  
STDPを活用した競合学習によって2つの表面粗さの異なるサンプルを教師なし学習で分類するためのモデル構築を行います．
