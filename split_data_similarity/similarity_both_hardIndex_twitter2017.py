import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import copy
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import sys
import torch
import pandas as pd
import gc
import numpy as np
import time

image_model = CLIPModel.from_pretrained("similarity/clip-vit-base-patch32")
image_processor = CLIPProcessor.from_pretrained("similarity/clip-vit-base-patch32")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
dataset = "100-3" # 修改
top_k = 50       # 修改

pd_test = pd.read_csv("./image_caption_ofa/twitter2017_process_caption_test.csv", sep='\t')
pd_train = pd.read_csv("./split_data_similarity/split_dataset/twitter2017/{}/train-{}.csv".format(dataset, dataset), sep='\t')
pd_test["similar_text_top50"] = ""
pd_test["similar_text_index_top50"] = ""
pd_test["similar_text_score_top50"] = ""

pd_test["similar_image_top50"] = ""
pd_test["similar_image_index_top50"] = ""
pd_test["similar_image_score_top50"] = ""

pd_test["similar_caption_top50"] = ""
pd_test["similar_caption_index_top50"] = ""
pd_test["similar_caption_score_top50"] = ""

# 同时考虑text和captain两部分的相似度排名，排名取平均，然后再取Top10
# pd_test["similar_both_top10"] = ""
pd_test["similar_tc_both_index_top50"] = ""
pd_test["similar_tc_both_score_top50"] = ""

pd_test["similar_ti_both_index_top50"] = ""
pd_test["similar_ti_both_score_top50"] = ""

def cos_sim(a, b):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def cal_features(pd_data, model, processor):
    feature_all = None
    print("******* calculate the image feature**********")
    length = len(pd_data)
    for i in tqdm(range(length)):
        root_image = "/data10T/caichenran/2023/MICL/dataset/mner/twitter2017_images/"
        test_image_id = pd_data.loc[i, "image_id"]
        test_image_path = root_image + test_image_id
        # image = Image.open(test_image_path).convert("RGB")
        try:
            # print("yes")
            image = Image.open(test_image_path).convert("RGB")
            test_image = Image.open(test_image_path)
            test_image = processor(images=test_image, return_tensors="pt")["pixel_values"]
            embedding_a = model.get_image_features(test_image)
        except(OSError, NameError):
            split_list = test_image_path.split('/')[:-1]
            head = '/'
            # 重新拼接路径
            for path in split_list:
                head = os.path.join(head, path)
            test_image_path = os.path.join(head, "17_06_4705.jpg")
            test_image = Image.open(test_image_path)
            test_image = processor(images=test_image, return_tensors="pt")["pixel_values"]
            embedding_a = model.get_image_features(test_image)
            embedding_a = embedding_a * 0
        
        if feature_all == None:
            feature_all = embedding_a
        else:
            feature_all = torch.cat((feature_all, embedding_a), 0)
    return feature_all


# text
test_text_sentence = pd_test["text"].to_list()
train_text_sentence = pd_train["text"].to_list()


test_image_embeddings = cal_features(pd_test, image_model, image_processor)
image_cos_sim = None
last_run = 0
print("calculate image similarity start")
for i in tqdm(range(last_run, len(pd_train))):
    root_image = "/data10T/caichenran/2023/MICL/dataset/mner/twitter2017_images/"
    test_image_id = pd_train.loc[i, "image_id"]
    test_image_path = root_image + test_image_id
    try:
        # print("yes")
        image = Image.open(test_image_path).convert("RGB")
        test_image = Image.open(test_image_path)
        test_image = image_processor(images=test_image, return_tensors="pt")["pixel_values"]
        embedding_a = image_model.get_image_features(test_image)
    except(OSError, NameError):
        split_list = test_image_path.split('/')[:-1]
        head = '/'
        # 重新拼接路径
        for path in split_list:
            head = os.path.join(head, path)
        test_image_path = os.path.join(head, "17_06_4705.jpg")
        test_image = Image.open(test_image_path)
        test_image = image_processor(images=test_image, return_tensors="pt")["pixel_values"]
        embedding_a = image_model.get_image_features(test_image)
        embedding_a = embedding_a * 0
    single_sim = cos_sim(embedding_a, test_image_embeddings)
    if image_cos_sim == None:
        image_cos_sim = single_sim
    else:
        image_cos_sim = torch.cat((image_cos_sim, single_sim), dim=0)

image_cos_sim = image_cos_sim.T
image_cos_sim = image_cos_sim.detach().numpy()
image_cos_sim = torch.tensor(image_cos_sim)
assert len(pd_train) == image_cos_sim.shape[1]
assert len(pd_test) == image_cos_sim.shape[0]

test_text_embeddings = model.encode(test_text_sentence)
train_text_embeddings = model.encode(train_text_sentence)
text_cos_sim = util.cos_sim(test_text_embeddings, train_text_embeddings)
del test_text_embeddings, train_text_embeddings

test_caption_sentence = pd_test["image_caption_ofa_large"].to_list()
train_caption_sentence = pd_train["image_caption_ofa_large"].to_list()

test_caption_embeddings = model.encode(test_caption_sentence)
train_caption_embeddings = model.encode(train_caption_sentence)
caption_cos_sim = util.cos_sim(test_caption_embeddings, train_caption_embeddings)
del test_caption_embeddings, train_caption_embeddings


#Add all pairs to a list with their cosine similarity score
print("calculate the similar")
for i in tqdm(range(text_cos_sim.shape[0])):
# for i in range(text_cos_sim.shape[0]):
    text_sentence_combinations, caption_sentence_combinations, image_combinations = [], [], []
    text_similar_text, text_similar_index, text_similar_score = [], [], []
    image_similar, image_similar_index, image_similar_score = [], [], []
    capt_similar_text, capt_similar_index, capt_similar_score = [], [], []
    tc_both_similar_text, tc_both_similar_index, tc_both_similar_score = [], [], []
    ti_both_similar_text, ti_both_similar_index, ti_both_similar_score = [], [], []
    
    for j in range(text_cos_sim.shape[1]):
        text_sentence_combinations.append([text_cos_sim[i][j], i, j])
        caption_sentence_combinations.append([caption_cos_sim[i][j], i, j])
        image_combinations.append([image_cos_sim[i][j], i, j])
    text_sentence_combinations_sort = sorted(text_sentence_combinations, key=lambda x: x[0], reverse=True)
    caption_sentence_combinations_sort = sorted(caption_sentence_combinations, key=lambda x: x[0], reverse=True)
    image_combinations_sort = sorted(image_combinations, key=lambda x: x[0], reverse=True)
    
    # text and caption
    text_sort_copy = copy.deepcopy(text_sentence_combinations_sort)
    capt_sort_copy = copy.deepcopy(caption_sentence_combinations_sort)
    image_sort_copy = copy.deepcopy(image_combinations_sort)
    # 添加排名
    for index in range(len(text_sort_copy)):
        text_sort_copy[index].append(index)
        capt_sort_copy[index].append(index)
        image_sort_copy[index].append(index)
    tc_both_comb = []
    ti_both_comb = []
    text_sort_copy = sorted(text_sort_copy, key=lambda x: x[2], reverse=False)
    capt_sort_copy = sorted(capt_sort_copy, key=lambda x: x[2], reverse=False)
    image_sort_copy = sorted(image_sort_copy, key=lambda x: x[2], reverse=False)
    for index in range(len(text_sort_copy)):
        tc_both_comb.append([text_sort_copy[index][0], capt_sort_copy[index][0], text_sort_copy[index][1], text_sort_copy[index][2], text_sort_copy[index][-1]+capt_sort_copy[index][-1]]) # 最后一个值，是两个排名之和，因为image caption相似度较大，不太合适用相似度
        ti_both_comb.append([text_sort_copy[index][0], image_sort_copy[index][0], text_sort_copy[index][1], text_sort_copy[index][2], text_sort_copy[index][-1]+image_sort_copy[index][-1]]) # 最后一个值，是文本相似度和图片相似度之和
    tc_both_comb_sort = sorted(tc_both_comb, key=lambda x: x[-1], reverse=False)
    ti_both_comb_sort = sorted(ti_both_comb, key=lambda x: x[-1], reverse=False) # hard

    for score_text, score_capt, row, col, both_score in tc_both_comb_sort[0:top_k]:
        # print("{} \t {} \t {:.4f}".format(test_text_sentence[row], train_text_sentence[col], text_cos_sim[row][col]))
        # both_similar_text.append(train_text_sentence[col])
        tc_both_similar_index.append(col)
        tc_both_similar_score.append(both_score)
    # pd_test.loc[i, "similar_both_top10"] = str(text_similar_text)
    pd_test.loc[i, "similar_tc_both_index_top50"] = str(tc_both_similar_index)
    pd_test.loc[i, "similar_ti_both_score_top50"] = str(tc_both_similar_score)
    
    for score_text, score_image, row, col, both_score in ti_both_comb_sort[0:top_k]:
        ti_both_similar_index.append(col)
        ti_both_similar_score.append(both_score/2)
    pd_test.loc[i, "similar_ti_both_index_top50"] = str(ti_both_similar_index)
    pd_test.loc[i, "similar_ti_both_score_top50"] = str(ti_both_similar_score)
        

    for score, row, col in text_sentence_combinations_sort[0:top_k]:
        # print("{} \t {} \t {:.4f}".format(test_text_sentence[row], train_text_sentence[col], text_cos_sim[row][col]))
        text_similar_text.append(train_text_sentence[col])
        text_similar_index.append(col)
        text_similar_score.append(text_cos_sim[row][col])
    pd_test.loc[i, "similar_text_top50"] = str(text_similar_text)
    pd_test.loc[i, "similar_text_index_top50"] = str(text_similar_index)
    pd_test.loc[i, "similar_text_score_top50"] = str(text_similar_score)

    for score, row, col in image_combinations_sort[0:top_k]:
        # print("{} \t {} \t {:.4f}".format(test_text_sentence[row], train_text_sentence[col], text_cos_sim[row][col]))
        image_similar.append(pd_train.loc[col, "image_id"])
        image_similar_index.append(col)
        image_similar_score.append(image_cos_sim[row][col])
    pd_test.loc[i, "similar_image_top50"] = str(image_similar)
    pd_test.loc[i, "similar_image_index_top50"] = str(image_similar_index)
    pd_test.loc[i, "similar_image_score_top50"] = str(image_similar_score)

    for score, row, col in caption_sentence_combinations_sort[0:top_k]:
        # print("{} \t {} \t {:.4f}".format(test_caption_sentence[row], train_caption_sentence[col], caption_cos_sim[row][col]))
        capt_similar_text.append(train_caption_sentence[col])
        capt_similar_index.append(col)
        capt_similar_score.append(caption_cos_sim[row][col])
    pd_test.loc[i, "similar_caption_top50"] = str(capt_similar_text)
    pd_test.loc[i, "similar_caption_index_top50"] = str(capt_similar_index)
    pd_test.loc[i, "similar_caption_score_top50"] = str(capt_similar_score)
    # if i == 100:
    #     break

root_path = "./split_data_similarity/result/twitter2017/{}/".format(dataset)
if not os.path.exists(root_path):
    os.makedirs(root_path)
test_target_path = root_path + "twitter2017_{}_process_caption_similar_image_text_hardindex_test.csv".format(dataset)
pd_test.to_csv(test_target_path, sep="\t", index=False)

train_target_path = root_path + "train-{}.csv".format(dataset)
pd_train.to_csv(train_target_path, sep="\t", index=False)