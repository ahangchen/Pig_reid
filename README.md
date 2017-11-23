## Rank Re-identification

### 模型

![](img/rank_model.png)

- 基础网络：ResNet50
- 输入：根据基础网络单模型计算相似度得到一个rank表，按表中的相似度选择一张待匹配图片A，备选图片B和C
- 输出：根据特征之间的欧氏距离计算A和B的二分类loss（是否为同一人），A和C的二分类loss，使用ranknet公式计算AB排序高于AC的概率，回归根据rank表计算得到的实际排序概率

### 硬件
- TITANX单卡（通过CUDA_VISIBLE_DEVICES指定使用哪块GPU）

### 代码说明
- baseline：ResNet50基础模型
  - [evaluate.py](baseline/evaluate.py)
    - extract_feature: 在测试集上计算特征
    - similarity_matrix: 根据特征计算特征之间的余弦相似度（使用GPU矩阵运算进行加速）
    - 在测试集上，调用test_predict计算rank表
    - 在训练集上，调用train_predict计算rank表
    - 使用map_rank_eval在Market1501上计算rank acc和map
    - 使用grid_result_eval在GRID上计算rank acc
  - [train.py](baseline/train.py)
    - 使用Market1501训练集数据预训练ResNet50基础网络
- pair: 双图二分类模型预训练
  - [pair_train.py](pair/pair_train.py)：双图预训练
    - pair_generator: 数据生成器，根据标签选择正样本和负样本
    - pair_model: 搭建二分类双输入模型
  - [eval](pretrian/eval.py)：各种模型的测试都在这里
    - 加载对应的模型
    - 调用baseline/evaluate.py中与数据集对应的函数进行测试
