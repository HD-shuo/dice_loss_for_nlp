#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: mrc_ner_dataset.py

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets.data_sampling import sample_positive_and_negative_by_ratio
import re
import pdb


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text, return_subtoken_start=False):
    """Returns tokenized answer spans that better match the annotated answer."""
    doc_tokens = [str(tmp) for tmp in doc_tokens]
    # 编码答案文本，且不对特殊字符进行编码
    answer_tokens = tokenizer.encode(orig_answer_text, add_special_tokens=False)
    # 将编码后的答案文本中间插入空格
    tok_answer_text = " ".join([str(tmp) for tmp in answer_tokens])
    # 遍历从input_start到input_end的所有起始位置
    for new_start in range(input_start, input_end + 1):
        # 逆序遍历从input_end到input_start的所有结束位置
        for new_end in range(input_end, new_start - 1, -1):
            # 根据当前起始位置与结束位置对doc_token的子序列进行空格分隔进行答案匹配
            text_span = " ".join(doc_tokens[new_start : (new_end+1)])
            # 如果当前子序列与答案文本完全匹配且不需要返回子标记的起始位置，返回匹配的起始与结束位置
            if text_span == tok_answer_text:
                if not return_subtoken_start:
                    return (new_start, new_end)
                # 将子序列标记索引转换回字符串
                tokens = tokenizer.convert_ids_to_tokens(doc_tokens[new_start: (new_end + 1)])
                # 如果最后一个标记不是分词的子标记，返回匹配的起始与结束位置
                if "##" not in tokens[-1]:
                    return (new_start, new_end)
                else:
                    for idx in range(len(tokens)-1, -1, -1):
                        if "##" not in tokens[idx]:
                            new_end = new_end - (len(tokens)-1 - idx)
                            return (new_start, new_end)

    return (input_start, input_end)


class WEBNERDataset(Dataset):
    """
    MRC NER Dataset
    Args:
        json_path: path to mrc-ner style json
        tokenizer: BertTokenizer
        # 取的最长长度为query+context
        max_length: int, max length of query+context
        possible_only: if True, only use possible samples that contain answer for the query/context
        is_chinese: is chinese dataset
    """
    def __init__(self, json_path, tokenizer: AutoTokenizer, max_length: int = 512, possible_only=False, is_chinese=True,
                 pad_to_maxlen=False, negative_sampling=True, prefix="train", data_sign="zh_onto", do_lower_case=False,
                 pred_answerable=True):
        self.all_data = json.load(open(json_path, encoding="utf-8"))
        self.tokenzier = tokenizer
        self.max_length = max_length
        self.do_lower_case = do_lower_case
        self.label2idx = {value:key for key, value in enumerate(WEBNERDataset.get_labels(data_sign))}

        if prefix == "train" and negative_sampling:
            neg_data_items = [x for x in self.all_data if not x["start_position"]]
            pos_data_items = [x for x in self.all_data if x["start_position"]]
            self.all_data = sample_positive_and_negative_by_ratio(pos_data_items, neg_data_items)
        elif prefix == "train" and possible_only:
            self.all_data = [
                x for x in self.all_data if x["start_position"]
            ]
        else:
            pass

        self.is_chinese = is_chinese
        self.pad_to_maxlen = pad_to_maxlen
        self.pred_answerable = pred_answerable

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        """
        Args:
            item: int, idx
        Returns:
            tokens: tokens of query + context, [seq_len]
            token_type_ids: token type ids, 0 for query, 1 for context, [seq_len]
            start_labels: start labels of NER in tokens, [seq_len]
            end_labels: end labelsof NER in tokens, [seq_len]
            label_mask: label mask, 1 for counting into loss, 0 for ignoring. shape of [seq_len]. 1 for no-subword context tokens. 0 for query tokens and [CLS] [SEP] tokens.
            match_labels: match labels, [seq_len, seq_len]
            sample_idx: sample id
            label_idx: label id

        """
        # 获取指定索引的样本数据
        data = self.all_data[item]
        tokenizer = self.tokenzier
        label_idx = torch.tensor(self.label2idx[data["entity_label"]], dtype=torch.long)

        query = data["query"]
        context = data["content"]

        # 处理实体位置，映射标签到分词后的文本
        start_positions = data["start_position"]
        end_positions = data["end_position"]

        # 更新每个item中的start_position与end_positions,主要是原始文本中去掉' '与'\n'的新位置
        for index, std_id in enumerate(start_positions):
            sen_count = len(re.findall(r'\s', context[:std_id]))
            if sen_count:
                start_positions[index] = std_id - sen_count
        
        for index, end_id in enumerate(end_positions):
            sen_count = len(re.findall(r'\s', context[:end_id]))
            if sen_count:
                end_positions[index] = end_id - sen_count - 1

        # 分词处理，将query与上下文进行分词
        # 如果is_chinese=true，则对context和query中的空格去除，将所有字符连接，再进行拼词
        if self.is_chinese:
            query = "".join(data["query"].strip().split())
            context = "".join(data["content"].strip().split())
        else:
            query = data["query"]
            context = data["content"]
        # 将分词后的文本转换为模型输入的张量
        # 这里将查询和上下文一同进行了编码，并在文本的开头与结尾添加特殊标记[CLS]与[SEP]
        # add_special_tokens=True：是否在文本的开头与结尾添加特殊标记
        # max_length:编码后的最大序列长度
        # return_overflowing_tokens: 是否返回超过max_length的部分
        # return_token_type_ids：是否返回token类型编码
        query_context_tokens = tokenizer.encode_plus(query, context,
            add_special_tokens=True,
            max_length=self.max_length,
            return_overflowing_tokens=True,
            return_token_type_ids=True)

        # 检查文本中是否存在填充标记，若存在则将其删除
        if tokenizer.pad_token_id in query_context_tokens["input_ids"]:
            non_padded_ids = query_context_tokens["input_ids"][: query_context_tokens["input_ids"].index(tokenizer.pad_token_id)]
        else:
            non_padded_ids = query_context_tokens["input_ids"]

        # 将query+context对应的token_ids转换回原先对应的文本
        non_pad_tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)
        # 查找第一个和最后一个[SEP]标识符位置，确定context与query分界点
        first_sep_token = non_pad_tokens.index("[SEP]")
        end_sep_token = len(non_pad_tokens) - 1
        new_start_positions = []
        new_end_positions = []
        if len(start_positions) != 0:
            for start_index, end_index in zip(start_positions, end_positions):
                if self.is_chinese:
                    # 如果是中文文本，答案文本跨度通过索引切片进行处理
                    answer_text_span = " ".join(context[start_index: end_index+1])
                else:
                    answer_text_span = " ".join(context.split(" ")[start_index: end_index+1])
                # 获得改进后的起始与结束位置
                new_start, new_end = _improve_answer_span(query_context_tokens["input_ids"], first_sep_token, end_sep_token, self.tokenzier, answer_text_span)
                new_start_positions.append(new_start)
                new_end_positions.append(new_end)
        else:
            new_start_positions = start_positions
            new_end_positions = end_positions

        # clip out-of-boundary entity positions.
        # 对新的起始与结束位置进行筛选，只保留了小于self.max_lenth的位置
        new_start_positions = [start_pos for start_pos in new_start_positions if start_pos < self.max_length]
        new_end_positions = [end_pos for end_pos in new_end_positions if end_pos < self.max_length]
        # 构造模型输入的各种张量
        tokens = query_context_tokens["input_ids"]
        token_type_ids = query_context_tokens['token_type_ids']
        # token_type_ids -> 0 for query tokens and 1 for context tokens.
        attention_mask = query_context_tokens['attention_mask']
        start_labels = [(1 if idx in new_start_positions else 0) for idx in range(len(tokens))]
        end_labels = [(1 if idx in new_end_positions else 0) for idx in range(len(tokens))]
        label_mask = [1 if token_type_ids[token_idx] == 1 else 0 for token_idx in range(len(tokens))]
        start_label_mask = label_mask.copy()
        end_label_mask = label_mask.copy()

        seq_len = len(tokens)
        # 生成实体匹配矩阵
        # 构造了一个大小为[seq_len, seq_len]的全0矩阵，若start与end存在，则把矩阵中对应的(start, end)设置为1
        match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)
        for start, end in zip(new_start_positions, new_end_positions):
            if start >= seq_len or end >= seq_len:
                continue
            match_labels[start, end] = 1

        # 根据模型需求，添加是否可回答的标签
        if self.pred_answerable:
            answerable_label = 1 if len(new_start_positions) != 0 else 0
            return [torch.tensor(tokens, dtype=torch.long),
                    torch.tensor(attention_mask, dtype=torch.long),
                    torch.tensor(token_type_ids, dtype=torch.long),
                    torch.tensor(start_labels, dtype=torch.long),
                    torch.tensor(end_labels, dtype=torch.long),
                    torch.tensor(start_label_mask, dtype=torch.long),
                    torch.tensor(end_label_mask, dtype=torch.long),
                    match_labels,
                    label_idx,
                    torch.tensor([answerable_label], dtype=torch.long)]

        return [torch.tensor(tokens, dtype=torch.long),
                torch.tensor(attention_mask, dtype=torch.long),
                torch.tensor(token_type_ids, dtype=torch.long),
                torch.tensor(start_labels, dtype=torch.long),
                torch.tensor(end_labels, dtype=torch.long),
                torch.tensor(start_label_mask, dtype=torch.long),
                torch.tensor(end_label_mask, dtype=torch.long),
                match_labels,
                label_idx]

    @classmethod
    def get_labels(cls, data_sign):
        """gets the list of labels for this data set."""
        if data_sign == "zh_onto":
            return ["GPE", "LOC", "PER", "ORG"]
        elif data_sign == "zh_msra":
            return ["NS", "NR", "NT"]
        elif data_sign == "en_onto":
            return ["LAW", "EVENT", "CARDINAL", "FAC", "TIME", "DATE", "ORDINAL", "ORG", "QUANTITY", \
                    "PERCENT", "WORK_OF_ART", "LOC", "LANGUAGE", "NORP", "MONEY", "PERSON", "GPE", "PRODUCT"]
        elif data_sign == "en_conll03":
            return ["ORG", "PER", "LOC", "MISC"]
        elif data_sign == "zh_web":
            return ["公司产品", "经营范围", "企业名称", "荣誉资质", "应用领域", "动态信息标题", 
                    "解决方案", "电话号码", "动态信息详情", "企业服务", "办公地址",  
                    "技术工艺", "企业简介","产品介绍", "关键人员", "人员职称", "人员规模", 
                    "服务介绍", "人员详情", "附属园区", "技术介绍", "附属园区地址", "中标信息标题", "中标信息详情"]
        return ["0", "1"]


