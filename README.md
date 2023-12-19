# ICL_for_FewMNER

## 数据集

`./datatset/mner`：原始MNER数据集，Twitter-2015, Twitter-2017


## 数据预处理

`./process_data/change_dataset_format_twitter2015/2017.ipynb`: 格式化数据集

`./obtain_image_caption/image_caption_OFA_twitter2015/2017.py`: 获得图片描述

`./split_data_similarity/split_dataset/split_twitter2015/2017.ipynb`: 采样数据集，|D|=10, |D|=50, |D|=100

## 计算图片和文本相似度rank和

`split_data_similarity/similarity_both_hardIndex_twitter2015/2017.py`: 计算图片和文本相似度rank和


## In-context Learning for Few-shot MNER

`multiMM_gpt3.5_twitter2015_50-1_shot-4.py`: In-context Learning for Few-shot MNER code



