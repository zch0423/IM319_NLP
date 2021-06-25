## 基于BERT的社会偏见语言识别模型

没有包含BERT等预训练模型，只包含了pickle导出的中间结果

## dataProcess

数据处理

- dataProcessing，处理原始数据和标签
- labelExtract，提取分类标注标签为npy文件

- bertDataProcess，处理BERT导出数据
- CNNandLabel，处理CNN需要的数据
- decomPCA，PCA

## data

Pickle部分中间结果保存

- labels，包含数据标注标注npy数组
- w2v，包含BERT处理后的结果，分为CLS/AVG，只保留了最后一层输出
- CNN，包含CNN模型部分用到的数据

## models

模型部分代码，一些中间结果和无用代码没有保存

- get_features.txt，BERT脚本见[Github-bert](https://github.com/google-research/bert)
- BERT_LR_SVM和bert_svm_nn，BERT模型
- CNN和cnn_train，CNN模型

## visual

可视化代码

