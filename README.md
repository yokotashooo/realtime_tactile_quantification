## 仕様
* `realtime_data_transfer`  
分散型ストリーミングプラットフォームであるApache kafkaを利用したデータ転送（実験環境）,本番は研究室のLinuxPCをサーバーとして起動し，データをストリーミング転送

  * `realtimeconsumer`  
    Apachekafkaへのデータの送信部  

  * `realtimeproducer`  
    Apachekafkaからデータの受信し，テスト分類処理

* `local_comment`  
社内メモの作成

  * `make-sample-reply`  
    届いた問い合わせに対して、返信例を社内メモに自動で作成する機能
