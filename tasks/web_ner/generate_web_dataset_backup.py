import json
import random
import copy
from tqdm import tqdm


def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def write_json_file(target_path, data):
    with open(target_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def split_train_and_valid(orign_file, target_train_file, target_valid_file):
    json_file = read_json_file(orign_file)
    length = len(json_file)
    valid_num = round(length*0.2)
    valid_json = write_json_file(target_valid_file, json_file[:valid_num])
    train_json = write_json_file(target_train_file, json_file[valid_num:])
    print("raw train and valid file created")
    return valid_json, train_json

def get_json_info(orign):
    orign_data = read_json_file(orign)
    all_texts = []
    all_start_ids = []
    all_end_ids = []
    all_entity_flags = []
    all_entitys = []
    for data in orign_data:
        text = data["data"]["text"]
        annos = data["annotations"]
        start_ids_raw = []
        end_ids_raw = []
        entitys_raw = []
        sen_lens = []
        sum_sen_lens = []
        entity_flags = []
        texts = []
        tmp_sum = 0

        for ann in annos:
            for lines in ann["result"]:
                if next(iter(lines.keys())) != "value":
                    break
                start_ids_raw.append(lines["value"]["start"])
                end_ids_raw.append(lines["value"]["end"])
                entitys_raw.append(lines["value"]["labels"])

        sentences = text.split("\n")
        for sentence in sentences:
            temp_len = len(sentence)
            tmp_sum = tmp_sum + temp_len + 1 
            sen_lens.append(temp_len)
            sum_sen_lens.append(tmp_sum)
            texts.append(sentence)
        entity_flags = [0] * len(sum_sen_lens)
        entitys = [None] * len(sum_sen_lens)
        start_ids = [None] * len(sum_sen_lens)
        end_ids = [None] * len(sum_sen_lens)

        for start_id in start_ids_raw:
            if start_id < sum_sen_lens[0]:
                entity_flags[0] += 1

        for start_index, start_id in enumerate(start_ids_raw):
            for slen_index, slen in enumerate(sum_sen_lens):
                if start_id >= slen and start_id < sum_sen_lens[slen_index + 1]:
                    entity_flags[slen_index + 1] += 1
                    start_id_new = start_id - slen
                    start_ids_raw[start_index] = start_id_new
                    end_id_new = end_ids_raw[start_index] - slen - 1
                    end_ids_raw[start_index] = end_id_new
                    break
        cn = 0 
        for index, flg in enumerate(entity_flags):
            if flg == 1:
                entitys[index] = entitys_raw[cn]
                start_ids[index] = start_ids_raw[cn]
                end_ids[index] = end_ids_raw[cn]
                cn += 1
        all_texts.append(texts)
        all_end_ids.append(end_ids)
        all_start_ids.append(start_ids)
        all_entitys.append(entitys)
        all_entity_flags.append(entity_flags)
    print("data analyse complete")
    return all_texts, all_start_ids, all_end_ids, all_entity_flags, all_entitys

def json_transform(orign, target):
    texts, start_ids, end_ids, entity_flags, entitys = get_json_info(orign)
    value_list = []
    values_list = []
    #新建一个空字典
    keys = ["context", "end_position", "entity_label", "impossible", "qas_id", "query", "span_position", "start_position"]
    dicts_list = []
    dic_len = 8
    cls = [['公司产品'], ['业务范围'], ['企业名称'], ['荣誉资质'], ['应用领域'], ['动态信息标题'],
            ['解决方案'], ['联系电话'], ['动态信息详情'], ['企业服务'], ['办公地址'], ['应用案例'],
              ['技术工艺'], ['企业简介'],['产品介绍'], ['关键人员'], ['人员职称'], ['人员规模'],
                ['服务介绍'], ['人员详情'], ['附属园区'], ['技术介绍'], ['附属园区地址'], ['中标信息标题'], ['中标信息详情']]
    dic_slice = [None] * dic_len

    for c in cls:
        dic_slice[2] = c
        tmp_slice = dic_slice.copy()
        value_list.append(tmp_slice)

    for tindex, txts in tqdm(enumerate(texts), total = len(texts), desc="Processing", ncols = 80):
        for index, text in enumerate(txts):
            for in_c, slice in enumerate(value_list):
                slice[0] = text
                slice[4] = f"{index}.{in_c}"
                slice[6] = []
                tmp_end = []
                tmp_start = []
                tmp_span = []
            if entity_flags[tindex][index] == 1:
                for slice in value_list:
                    if slice[2] == entitys[tindex][index]:
                        slice[3] = "False"
                        tmp_pair = str(start_ids[tindex][index]) + ';' + str(end_ids[tindex][index])
                        tmp_end.append(end_ids[tindex][index])
                        tmp_start.append(start_ids[tindex][index])
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
                    slice[3] = "True"
                    slice[1] = []
                    slice[6] = []
                    slice[7] = []
            tmp_value_list = copy.deepcopy(value_list)
            values_list.append(tmp_value_list)

    for v_list in values_list:
        dict_list = [{key: value for key, value in zip(keys, values)} for values in v_list]
        dicts_list.append(dict_list)
    
    write_json_file(target, dicts_list)

if __name__ == "__main__":
    source_file = '/home/daixingshuo/dice_loss_for_NLP/dataset/web/7.18merge.json'
    valid_json = '/home/daixingshuo/dice_loss_for_NLP/dataset/web/web_valid.json'
    train_json = '/home/daixingshuo/dice_loss_for_NLP/dataset/web/web_train.json'
    target_train_file = '/home/daixingshuo/dice_loss_for_NLP/dataset/web/web-ner.train'
    target_valid_file = '/home/daixingshuo/dice_loss_for_NLP/dataset/web/web-ner.dev'
    # split_train_and_valid(source_file, train_json, valid_json)
    #json_transform(valid_json, target_valid_file)
    #print("valid file created!")
    json_transform(train_json, target_train_file)
    print("train file created!")