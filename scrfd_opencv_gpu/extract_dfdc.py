# -*- coding: utf-8 -*-
# @Time    : 2023/4/13 17:32
# @Author  : yblir
# @File    : extract_dfdc.py
# explain  : 
# =======================================================
import json
import random


def get_related_mp4(json_path):
    json_data = json.load(open(json_path, 'r'))
    real_video_list = []
    fake_video_list = []
    for key, value in json_data.items():
        # print(key,value)
        label = value['label']
        if label == 'REAL':
            real_video_list.append(key)

    return real_video_list


def get_related_mp4_new(json_path):
    json_data = json.load(open(json_path, 'r'))
    real_video_dic = {}
    output = []
    # 提取所有real视频
    for key, value in json_data.items():
        if value['label'] == 'REAL':
            real_video_dic[key] = []

    # 提取所有real对应的fake视频
    for key, value in json_data.items():
        # key == "ijptktlyfr.mp4"
        if value['label'] == 'FAKE':
            real_video_dic[value['original']].append(key)

    # 每个real视频对应2个fake视频
    # count = 0
    for key, value in real_video_dic.items():
        # count += len(value)
        print(len(value), end=" | ")
        random.shuffle(value)
        output.append(key)
        output.extend(value[:2])
        # output.append(value)
    # print("=== ", count)
    return output


if __name__ == '__main__':
    json_path = r'D:\Downloads\dfdc_train_part_02\dfdc_train_part_2\metadata.json'
    # real_video_list, fake_video_list = get_related_mp4(json_path)
    res = get_related_mp4_new(json_path)

    print(len(res))
    # print(len(fake_video_list))
