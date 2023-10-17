import json
import copy
from tqdm import tqdm


def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def write_json_file(target_path, data):
    with open(target_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def split_train_and_valid(orign_file, target_train_file, target_valid_file, target_test_file):
    json_file = read_json_file(orign_file)
    #length = len(json_file)
    #valid_num = round(length*0.2)
    valid_json = write_json_file(target_valid_file, json_file[:7200])
    print("valid file created!")
    train_json = write_json_file(target_test_file, json_file[7200:9600])
    print("text file created!")
    test_json = write_json_file(target_train_file, json_file[9600:])
    print("train and valid file created")
    return valid_json, train_json, test_json

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
        start_ids = []
        end_ids = []
        entitys = []

        for ann in annos:
            for lines in ann["result"]:
                if next(iter(lines.keys())) != "value":
                    break
                start_ids.append(lines["value"]["start"])
                end_ids.append(lines["value"]["end"])
                entitys.append(lines["value"]["labels"])
        all_texts.append(text)
        all_end_ids.append(end_ids)
        all_start_ids.append(start_ids)
        all_entitys.append(entitys)
    print("data analyse complete")
    return all_texts, all_start_ids, all_end_ids, all_entitys

def json_transform(orign, target):
    texts, start_ids, end_ids, entitys = get_json_info(orign)
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
    query = ['products of the company','business scope','the name of company','Honorary qualifications','application area',
             'Dynamic Information Title','solution','tel number','Dynamic information details','company services','the adress of company',
             'Application Cases','technology','Company Profile','introduction of products','Key personnel','Personnel Title',
             'staff size','services introduction','Personnel details','Affiliated park','Technology introduction','Affiliated park address',
             'Title of bid winning information','Details of bid winning information']
    dic_slice = [None] * dic_len

    for ind, c in enumerate(cls):
        dic_slice[2] = c
        dic_slice[5] = query[ind]
        tmp_slice = dic_slice.copy()
        value_list.append(tmp_slice)

    for tindex, txts in tqdm(enumerate(texts), total = len(texts), desc="Processing", ncols = 80):
        for in_c, slice in enumerate(value_list):
            slice[0] = txts
            slice[4] = f"{tindex}.{in_c}"
            slice[6] = []
        tmp_end = []
        tmp_start = []
        tmp_span = []
        
        for slice in value_list:
            for index, ent in enumerate(entitys[tindex]):
                if slice[2] == ent:
                    slice[3] = "False"
                    tmp_pair = str(start_ids[tindex][index]) + ';' + str(end_ids[tindex][index])
                    tmp_end.append(end_ids[tindex][index])
                    tmp_start.append(start_ids[tindex][index])
                    tmp_span.append(tmp_pair)
                    slice[1] = tmp_end
                    slice[7] = tmp_start
                    slice[6] = tmp_span
                if slice[2] != ent:
                    if slice[1] == None:
                        slice[3] = "True"
                        slice[1] = []
                        slice[7] = []
                        slice[6] = []
                    else:
                        continue
            tmp_end = []
            tmp_start = []
            tmp_span = []
        tmp_value_list = copy.deepcopy(value_list)
        values_list.append(tmp_value_list)

    for v_list in values_list:
        dict_list = [{key: value for key, value in zip(keys, values)} for values in v_list]
        dicts_list.append(dict_list)
    
    result_list = []
    for sub_dict in dicts_list:
        for dic in sub_dict:
            result_list.append(dic)
    """
    with open(target, 'w') as f:
        for dicts in dicts_list:
            json.dump(dicts, f, indent=2, ensure_ascii=False)
    """
    write_json_file(target, result_list)

if __name__ == "__main__":
    source_file = '/home/daixingshuo/dice_loss_for_NLP/dataset/web/7.18merge.json'
    valid_json = '/home/daixingshuo/dice_loss_for_NLP/dataset/web/web_valid.json'
    train_json = '/home/daixingshuo/dice_loss_for_NLP/dataset/web/web_train.json'
    target_train_file = '/home/daixingshuo/dice_loss_for_NLP/dataset/web/web-ner.train'
    target_valid_file = '/home/daixingshuo/dice_loss_for_NLP/dataset/web/web-ner.dev'
    target_test_file = '/home/daixingshuo/dice_loss_for_NLP/dataset/web/web-ner.test'
    origin_file = '/home/daixingshuo/dice_loss_for_NLP/dataset/web/8.8web_line_ner_annotation.json'
    split_train_and_valid(origin_file, target_train_file, target_valid_file, target_test_file)
    """
    json_transform(valid_json, target_valid_file)
    print("valid file created!")
    json_transform(train_json, target_train_file)
    print("train file created!")
    """