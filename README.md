## 仕様
* `realtime_data_transfer`  
分散型ストリーミングプラットフォームであるApache kafkaを利用したデータ転送（実験環境）,本番は研究室のLinuxPCをサーバーとして起動し，データをストリーミング転送

  * `realtimeconsumer`  
    Apachekafkaへのデータの送信する機能

  * `realtimeproducer`  
    Apachekafkaからデータの受信し，テスト分類処理する機能

* `realtime_spike_encoding`  
振動データを取得し，スパイク化するために必要なコード群

  * `data_collect.py`  
    振動がある閾値を超えたらデータを2秒間収集する機能
 
  * `realtime_encoding.py`  
    取得したセンサの振動をwindowsでリアルタイムスパイク化する機能(実験)

  * `rasp.py`  
    取得したセンサの振動をraspberrypiでリアルタイムスパイク化し，Apache kafakのサーバーにリアルタイムで転送する機能(本番)

* `2classification.ipynb`  
STDPを活用した競合学習によって2つの表面粗さの異なるサンプルを教師なし学習で分類するためのモデル構築を行う機能
