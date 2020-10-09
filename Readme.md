## 百度常规赛的赛题
利用10万多条技师与用户的多轮对话与诊断建议报告（其中训练集82943条记录，测试集20000条记录），建立模型，基于对话文本、用户问题、车型与车系，来预测输出诊断建议报告文本。<br>
使用训练好的模型，根据测试集数据输出对应建议报告的结果文件。

代码结构
+ datasets (数据集)
    + AutoMaster_TrainSet.csv 训练集
    + AutoMaster_TestSet.csv  测试集
+ seq2seq_tf2 (seq2seq 模型结构)
    + bin/
    + encoders/
    + decoders/  # 使用attention机制
    + models/
    + batcher.py
    + test_helper.py  # 比较greedy search 和 beam search 两种预测效果
    + train_eval_test.py
    + train_helper.py
+ seq2seq_pgn_tf2 (seq2seq + pgn模型结构)
    + 在seq2seq_tf2基础上搭建PGN网络，使用coverage策略，beam search策略
    + 增加了utils/ # losses + calc_final_dist
+ seq2seq_transformer_pgn_tf2 (seq2seq + transformer + pgn模型结构)
    + seq2seq_pgn_tf2 基础上，使用transformer单元结构替换了原来的bigru/bilstm结构
    + 增加layers/  # 放transformer内部的网络层
    + 增加schedules/  # 使用warmup函数改进学习率
+ seq2seq_bertsum (bert抽提式摘要模型结构)
    + 尝试...
+ utils 工具包
    + preprocess  清洗数据
    + data_loader  生成训练集，和测试集
    + build_w2v  word2vec训练词向量
    + data_utils  从训练集中分出验证集
    + data_utils  工具函数


运行步骤:
1. 依次运行utils目录下的 preprocess.py、data_loader.py、build_w2v.py、data_utils.py文件，预处理数据，构建数据集和词向量。
2. 每个模型的bin目录下都有main.py文件，配置好模型的参数，选择训练、验证、测试三种模式，运行即可。
