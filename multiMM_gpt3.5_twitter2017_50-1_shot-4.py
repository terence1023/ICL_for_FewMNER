import openai
import os
import re
import ast
import pandas as pd
import json
from tqdm import tqdm 
import time

# need to change
openai.api_key = "sk-XXX"

dataset = "50-1"
top_k = 4
last_run = 0

root_path = "split_data_similarity/result/twitter2017/"
test_file = "twitter2017_{}_process_caption_similar_image_text_hardindex_test.csv".format(dataset)
train_file = "train-{}.csv".format(dataset)

pd_test_path = root_path + dataset + "/" + test_file
pd_train_path = root_path + dataset + "/" + train_file

# read test dataset
pd_test = pd.read_csv(pd_test_path, sep='\t')
pd_train = pd.read_csv(pd_train_path, sep='\t')
chat_predict = pd.DataFrame(columns=["text", "person", "organization", "location", "miscellaneous"])
chat_list = []

SYSTEM_PROMPT = "You are a smart and intelligent Multimodal Named Entity Recognition (MNER) system. I will provide you the definition of the entities you need to extract, the sentence from where your extract the entities, the image description from image associated with sentence and the output format with examples."

USER_PROMPT_1 = "Are you clear about your role?"

ASSISTANT_PROMPT_1 = "Sure, I'm ready to help you with your MNER task. Please provide me with the necessary information to get started."

def openai_chat_completion_response(final_prompt):
    try:
        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_1},
                    {"role": "assistant", "content": ASSISTANT_PROMPT_1},
                    {"role": "user", "content": final_prompt}
                ]
            )
        return response['choices'][0]['message']['content'].strip(" \n")
    except Exception as e:
        time.sleep(3)
        return openai_chat_completion_response(final_prompt)


for i in tqdm(range(last_run, len(pd_test))):
    my_sentence = pd_test.loc[i, "text"]
    my_caption = str(pd_test.loc[i, "image_caption_ofa_large"]) + "."
    rank = pd_test.loc[i, "similar_ti_both_index_top50"] # 使用image和text综合排序
    rank_list = rank[1:-1].split(',')
    rank_list = [int(x.strip()) for x in rank_list][:top_k]
    sentences = []
    outputs = {}
    images = []
    for index in range(top_k-1,-1,-1):
        number = rank_list[index]
        train_text = pd_train.loc[number, "text"]
        image_caption = pd_train.loc[number, "image_caption_ofa_large"] + "."
        sentences.append(train_text)	
        person = pd_train.loc[number, "person"]
        if person == "[]":
            person = "['None']"
        location = pd_train.loc[number, "location"]
        if location == "[]":
            location = "['None']"
        organization = pd_train.loc[number, "organization"]
        if organization == "[]":
            organization = "['None']"
        miscellaneous = pd_train.loc[number, "miscellaneous"]
        if miscellaneous == "[]":
            miscellaneous = "['None']"
        single_output = {}
        single_output["PERSON"] = person
        single_output["ORGANIZATION"] = organization
        single_output["LOCATION"] = location
        single_output["MISCELLANEOUS"] = miscellaneous
        outputs[top_k-index-1] = single_output
        images.append(image_caption)

    GUIDELINES_PROMPT = (
        "Entity Definition:\n"
        "1. PERSON: Short name or full name of a person from any geographic regions.\n"
        "2. ORGANIZATION: An organized group of people with a particular purpose, such as a business or a government department.\n"
        "3. LOCATION: Name of any geographic location, like cities, countries, continents, districts etc.\n"
        "4. MISCELLANEOUS: Name entities that do not belong to the previous three groups PERSON, ORGANIZATION, and LOCATION.\n"
        "\n"
        "Output Format:\n"
        "{{'PERSON': [list of entities present], 'ORGANIZATION': [list of entities present], 'LOCATION': [list of entities present], 'MISCELLANEOUS': [list of entities present]}}\n"
        "If no entities are presented in any categories keep it None\n"
        "\n"
        "Examples:\n"
        "\n"
        "1. Image description: {}\n"
        "Sentence: {}\n"
        "Output: {}\n"
        "\n"
        "2. Image description: {}\n"
        "Sentence: {}\n"
        "Output: {}\n"
        "\n"
        "3. Image description: {}\n"
        "Sentence: {}\n"
        "Output: {}\n"
        "\n"
        "4. Image description: {}\n"
        "Sentence: {}\n"
        "Output: {}\n"
        "\n"
        "5. Image description: {}\n"
        "Sentence: {}\n"
        "Output: "
    ).format(images[0], sentences[0], outputs[0], images[1], sentences[1], outputs[1], images[2], sentences[2], outputs[2], images[3], sentences[3], outputs[3], my_caption, my_sentence)
    
    GUIDELINES_PROMPT = GUIDELINES_PROMPT.replace("\"", "")
    ners = openai_chat_completion_response(GUIDELINES_PROMPT)
    try:
        ners_dictionary = ast.literal_eval(ners)
        chat_predict.loc[i, "text"] = my_sentence
        for entity_type, entity_list in ners_dictionary.items():
            entity_list = list(set(entity_list))
            chat_predict.loc[i, entity_type.lower()] = entity_list
    except(OSError, NameError, SyntaxError):
        chat_predict.loc[i, "text"] = my_sentence
        pass
    chat_predict.to_csv("multiMM_gpt3.5_twitter2017_50-1_shot-4_predict_test.csv", sep='\t', index=False)