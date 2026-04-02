import json
import math
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import os 
# os.environ["TRANSFORMERS_OFFLINE"] = "1" # remove this line when downloading fresh
import numpy as np
import pandas as pd

import random

def load_model_tokenizer(model_name, device, dtype=torch.float32, device_map=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_attentions=True,
        dtype=dtype,
        local_files_only=False,  # set True when the model is already downloaded
        device_map=device_map,
    )
    if device_map is None:
        model.to(device)
    model.eval()
    return tokenizer, model

class PromptUtils:
    def __init__(self, tokenizer, doc_ids, dict_all_docs):
        self.dict_doc_name_id = {key:idx for idx, key in enumerate(doc_ids)}
        self.tokenizer = tokenizer
        self.prompt_seperator = " \n\n"
        user_header = '<|start_header_id|>user<|end_header_id|>'
        asst_header = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        self.item_instruction = f" Here are all the available tools:"
        self.prompt_prefix = user_header + self.item_instruction
        self.prompt_suffix = asst_header
        self.prompt_prefix_length = len(tokenizer(self.prompt_prefix, add_special_tokens=False).input_ids)
        self.prompt_suffix_length = len(tokenizer(self.prompt_suffix, add_special_tokens=False).input_ids)
        
        self.doc_text = lambda idx, doc_name, doc_info: f"tool_id: {doc_name}\ntool description: {doc_info}"
        self.add_text1 = f"Now, please output ONLY the correct tool_id for the query below."

        (
            self.all_docs_info_string, 
            self.doc_names_str, 
            self.doc_lengths,
            self.doc_spans
        ) = self.create_doc_pool_string(doc_ids, dict_all_docs)
        self.add_text1_length = len(tokenizer(self.add_text1, add_special_tokens=False).input_ids)

    
    def create_prompt(self, query):
        query_prompt = f"Query: {query}"+ "\nCorrect tool_id:"
        prompt = self.prompt_prefix + \
                self.all_docs_info_string + \
                self.prompt_seperator + \
                self.add_text1 + \
                self.prompt_seperator + \
                query_prompt + \
                self.prompt_suffix
        return prompt
        

    def create_doc_pool_string(self, shuffled_keys, all_docs):
        doc_lengths = []
        doc_list_str = []
        map_docname_id, map_id_docname = {}, {}
        all_schemas = ""
        doc_spans = []
        doc_st_index = self.prompt_prefix_length + 1 # inlcudes " \n\n"
        for ix, key in enumerate(shuffled_keys):
            value = all_docs[key]
            doc_list_str.append(key)
            text = self.prompt_seperator
            doc_text = self.doc_text(idx=self.dict_doc_name_id[key] + 1, doc_name=key, doc_info=value).strip()
            doc_text_len = len(self.tokenizer(doc_text, add_special_tokens=False).input_ids)
            text += doc_text
            doc_spans.append((doc_st_index, doc_st_index + doc_text_len))
            doc_st_index =  doc_st_index + 1 + doc_text_len
            doc_lengths.append(doc_text_len)
            all_schemas += text
            if ix == len(shuffled_keys)-1:
                end_of_docs_index = doc_st_index
        doc_list_str = ", ".join(doc_list_str)    
        return all_schemas, doc_list_str, doc_lengths, doc_spans
    
    

def get_queries_and_items_check():
    tool_path = "/scratch/deekshak/datasets/MetaTool/dataset/data/all_clean_data.csv"   
    tool_desc_path = "/scratch/deekshak/datasets/MetaTool/dataset/plugin_des.json"
    df =  pd.read_csv(tool_path)
    with open(tool_desc_path) as f:
        dbs = json.load(f)
    queries = []
    map_tool_count = {key: 0 for key in dbs}
    for idx in range(len(df)):
        row = df.iloc[idx]
        queries.append({
            "text": row["Query"],
            "gold_tool_name": row["Tool"],
            "qid": idx
            }
        )
        map_tool_count[row["Tool"]] += 1
    
    tools100 = sorted(map_tool_count.items(), key= lambda x: x[1], reverse=True)[:100]
    tools100 = [i[0] for i in tools100]
    queries_filtered = [i for i in queries if i["gold_tool_name"] in tools100]
    random.shuffle(queries_filtered)
    dbs_filtered = {i:dbs[i] for i in dbs if i in tools100}
    with open("data/test_queries.json", "w") as f: json.dump(queries_filtered[:5000], f)
    with open("data/train_queries.json", "w") as f: json.dump(queries_filtered[5000: 6500], f)
    with open("data/tools.json", "w") as f: json.dump(dbs_filtered, f)
    return queries_filtered, dbs_filtered


def get_queries_and_items():
    with open("data/test_queries.json", "r") as f: test_queries = json.load(f)
    with open("data/train_queries.json", "r") as f: train_queries  = json.load(f)
    with open("data/tools.json", "r") as f: tools = json.load(f)
    return train_queries, test_queries, tools


def load_dense_encoder(model_name, device, dtype=torch.float32):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
    model = AutoModel.from_pretrained(model_name, dtype=dtype, local_files_only=False)
    model.to(device)
    model.eval()
    return tokenizer, model


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    masked = last_hidden_state * mask
    denom = mask.sum(dim=1).clamp(min=1.0)
    return masked.sum(dim=1) / denom


def _l2_normalize(x, eps=1e-12):
    return x / torch.clamp(x.norm(p=2, dim=-1, keepdim=True), min=eps)


def encode_texts(texts, tokenizer, model, device, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            pooled = _mean_pool(outputs.last_hidden_state, inputs.attention_mask)
            embeddings.append(_l2_normalize(pooled).cpu())
    return torch.cat(embeddings, dim=0)


def compute_recall_at_k(ranked_ids, gold_id, k_list=(1, 5)):
    recalls = {}
    for k in k_list:
        recalls[k] = 1.0 if gold_id in ranked_ids[:k] else 0.0
    return recalls


def simple_tokenize(text):
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


class BM25Okapi:
    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens = corpus_tokens
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0.0
        self._initialize()

    def _initialize(self):
        df = {}
        total_len = 0
        for tokens in self.corpus_tokens:
            freqs = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            self.doc_freqs.append(freqs)
            self.doc_len.append(len(tokens))
            total_len += len(tokens)
            for t in freqs.keys():
                df[t] = df.get(t, 0) + 1
        self.avgdl = total_len / max(len(self.corpus_tokens), 1)
        total_docs = len(self.corpus_tokens)
        for t, freq in df.items():
            self.idf[t] = math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query_tokens):
        scores = [0.0] * len(self.corpus_tokens)
        for idx, freqs in enumerate(self.doc_freqs):
            score = 0.0
            dl = self.doc_len[idx]
            for t in query_tokens:
                if t not in freqs:
                    continue
                tf = freqs[t]
                idf = self.idf.get(t, 0.0)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * (tf * (self.k1 + 1)) / denom
            scores[idx] = score
        return scores


def build_bm25_index(tool_texts):
    corpus_tokens = [simple_tokenize(t) for t in tool_texts]
    return BM25Okapi(corpus_tokens)


def bm25_rank(query, bm25):
    query_tokens = simple_tokenize(query)
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked, scores