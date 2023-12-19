import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import pandas as pd
from PIL import Image
from tqdm import tqdm

def pred_caption(img_captioning, pd_data):
    pd_data["image_caption"] = ""
    pd_data["no_image"] = "No"
    for i in tqdm(range(len(pd_data))):
    # for i in range(len(pd_data)):
        root_path = "./dataset/mner/twitter2015_images/"
        image_id = pd_data.loc[i, "image_id"]
        image_path = root_path + image_id
        # 存在一些不可读的图片
        try:
            image = Image.open(image_path).convert("RGB")
        except(OSError, NameError):
            split_list = image_path.split('/')[:-1]
            head = ''
            # 重新拼接路径
            for path in split_list:
                head = os.path.join(head, path)
            image_path = os.path.join(head, "17_06_4705.jpg")
            pd_data.loc[i, "no_image"] = "Yes"
            
        result = img_captioning(image_path)
        image_caption = result[OutputKeys.CAPTION]
        pd_data.loc[i, "image_caption_ofa_large"] = image_caption
    return pd_data
    
if __name__ == "__main__":
    img_captioning = pipeline(Tasks.image_captioning, model='damo/ofa_image-caption_coco_large_en', model_revision='v1.0.1')
    print("******Load twitter2015_process_train.csv******")
    pd_train = pd.read_csv("./process_data/twitter2015_process_train.csv", sep="\t")
    pd_train_caption = pred_caption(img_captioning, pd_train)
    pd_train_caption.to_csv("twitter2015_process_caption_train.csv", sep='\t', index=False)
    print("******Finish Caption twitter2015_process_train.csv******")
    
    print("******Load twitter2015_process_dev.csv******")
    pd_dev = pd.read_csv("./process_data/twitter2015_process_dev.csv", sep='\t')
    pd_dev_caption = pred_caption(img_captioning, pd_dev)
    pd_dev_caption.to_csv("twitter2015_process_caption_dev.csv", sep='\t', index=False)
    print("******Finish Caption twitter2015_process_dev.csv******")
    
    print("******Load twitter2015_process_test.csv******")
    pd_test = pd.read_csv("./process_data/twitter2015_process_test.csv", sep='\t')
    pd_test_caption = pred_caption(img_captioning, pd_test)
    pd_test_caption.to_csv("twitter2015_process_caption_test.csv", sep='\t', index=False)
    print("******Finish Caption twitter2015_process_test.csv******")
    