"""
# 拆分标注集脚本
import copy
import json

def write_json_file(target_path, data):
    with open(target_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

dicts_list = []
dic_len = 8
cls = ['企业名称', '联系电话', '动态信息标题']
keys = ["context", "end_position", "entity_label", "impossible", "qas_id", "query", "span_position", "start_position"]
start_ids = [14,4,None]
end_ids = [28,17,None]
dic_slice = [None] * dic_len
value_list = []
values_list = []
tmp_slice = []
tmp_value_list = []
entity_flags = [1,1,0]
entitys = ['企业名称','企业名称', None]
texts = ['公司新闻 - 新闻动态 - 芜湖通和汽车流体系统有限公司', '欢迎来到芜湖通和汽车流体系统有限公司', '中文']

for in_c, c in enumerate(cls):
    dic_slice[2] = c
    tmp_slice = dic_slice.copy()
    value_list.append(tmp_slice)

for index, text in enumerate(texts):
    for in_c, slice in enumerate(value_list):
        slice[0] = text
        slice[4] = f"{index}.{in_c}"
        slice[6] = []
        tmp_end = []
        tmp_start = []
        tmp_span = []
    if entity_flags[index] == 1:
        for slice in value_list:
            if slice[2] == entitys[index]:
                slice[3] = "False"
                tmp_pair = str(start_ids[index]) + ';' + str(end_ids[index])
                tmp_end.append(end_ids[index])
                tmp_start.append(start_ids[index])
                tmp_span.append(tmp_pair)
                slice[1] = tmp_end
                slice[7] = tmp_start
                slice[6] = tmp_span
            else:
                slice[3] = "True"
                slice[1] = []
                slice[7] = []
    else:
        for slice in value_list:
            slice[3] = None
            slice[1] = []
            slice[6] = []
            slice[7] = []
    tmp_value_list = copy.deepcopy(value_list)
    values_list.append(tmp_value_list)   

for v_list in values_list:
            dict_list = [{key: value for key, value in zip(keys, values)} for values in v_list]
            dicts_list.append(dict_list)

target = "/home/daixingshuo/dice_loss_for_NLP/targetymp.json"
write_json_file(target, dicts_list)
"""

"""
计算文本类别种类及对应数量
"""
"""
import json
from collections import Counter

def get_labels(json_file, target_label):
    with open(json_file, 'r') as f:
        datas = json.load(f)

    labels = []
    for data in datas:
        annos = data["annotations"]
        for ann in annos:
            for lines in ann["result"]:
                if next(iter(lines.keys())) != "value":
                    break
                label_tuple = tuple(lines['value']['labels'])
                labels.append(label_tuple)
    label_counts = Counter(labels)
    return label_counts

if __name__ == "__main__":
    json_file = "/home/daixingshuo/dice_loss_for_NLP/dataset/web/7.18merge.json"
    target_label = "labels"
    label_counts = get_labels(json_file, target_label)
    print(label_counts)
"""

"""
核对生成的标注文件
"""

import json

def read_json(json_file):
    with open(json_file, 'r') as f:
        datas = json.load(f)
    return datas

def write_json_file(target_path, data):
    with open(target_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def re_sort_json(orign):
    orign_data = read_json(orign)
    ann_result = []
    stids = []
    annos = orign_data["annotations"]
    for ann in annos:
        for lines in ann["result"]:
            if next(iter(lines.keys())) != "value":
                break
            ann_result.append(lines)
            stids.append(lines["value"]["start"])
        # 为start_id建立索引
        sorted_start_ids = sorted(enumerate(stids), key=lambda x: x[1])
        sorted_stid_v = [item[1] for item in sorted_start_ids]
        sorted_stid_i = [item[1] for item in sorted_start_ids]
        sorted_result = sorted(ann_result, key=lambda x: x['start'])
        ann['result'].update(sorted_result)
    return orign_data
    
if __name__ == "__main__": 
    raw_path = '/home/daixingshuo/dice_loss_for_NLP/dataset/web/web_valid.json'
    file_path = "/home/daixingshuo/dice_loss_for_NLP/dataset/web/web-ner.dev"
    target = "/home/daixingshuo/dice_loss_for_NLP/dataset/web/for_debug/tmp.json"
    sample_path = '/home/daixingshuo/dice_loss_for_NLP/dataset/web/for_debug/sample.json'
    new_path = '/home/daixingshuo/dice_loss_for_NLP/dataset/web/8.8web_line_ner_annotation.json'
    test_path = '/home/daixingshuo/dice_loss_for_NLP/dataset/web/for_debug/test.json'
    #pdata = re_sort_json(sample_path)
    #write_json_file(std_test_path, pdata)
    datas = read_json(new_path)
    write_json_file(test_path, datas[1])