import torch
import gc
from tqdm import tqdm
from utils import PromptUtils
import random 

def select_retrieval_heads(train_queries, model, tokenizer, tools, device, max_heads=20):
    # TODO 3: Head selection

    num_layers = model.config.num_hidden_layers
    num_heads  = model.config.num_attention_heads

    # accumulate MRR score per head  [num_layers, num_heads]
    target_device = model.device
    head_scores = torch.zeros(num_layers, num_heads, device=target_device)

    for qix in tqdm(range(len(train_queries)), desc="Selecting heads"):

        sample         = train_queries[qix]
        question       = sample["text"]
        gold_tool_name = sample["gold_tool_name"]

        tool_ids = list(tools.keys())
        random.shuffle(tool_ids)

        putils = PromptUtils(
            tokenizer=tokenizer,
            doc_ids=tool_ids,
            dict_all_docs=tools,
        )
        item_spans     = putils.doc_spans
        doc_lengths    = putils.doc_lengths
        map_docname_id = putils.dict_doc_name_id

        gold_tool_id = map_docname_id[gold_tool_name]

        prompt = putils.create_prompt(query=question)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = inputs.to(target_device)

        input_ids = inputs.input_ids[0]

        with torch.no_grad():
            attentions = model(**inputs).attentions

        # ---------------------------------------------------------------
        # Locate the query token span in the full prompt
        # prompt structure: prefix | docs | sep | add_text1 | sep | query_prompt | suffix
        # query_prompt = "Query: {question}\nCorrect tool_id:"
        # ---------------------------------------------------------------
        query_prompt    = f"Query: {question}\nCorrect tool_id:"
        query_token_len = len(
            tokenizer(query_prompt, add_special_tokens=False).input_ids
        )
        total_len   = input_ids.shape[0]
        query_end   = total_len - putils.prompt_suffix_length
        query_start = query_end - query_token_len

        # ---------------------------------------------------------------
        # Score every (layer, head) by reciprocal rank of gold tool
        # ---------------------------------------------------------------
        num_docs = len(item_spans)

        for layer_idx in range(num_layers):
            # attentions[layer_idx]: [1, num_heads, N, N]
            layer_attn = attentions[layer_idx][0].to(target_device)   # [num_heads, N, N]

            for head_idx in range(num_heads):
                head_attn = layer_attn[head_idx]    # [N, N]

                # Compute one score per doc: total attention from query → doc tokens
                doc_scores_head = torch.zeros(num_docs, device=target_device)
                for doc_idx, (ds, de) in enumerate(item_spans):
                    doc_scores_head[doc_idx] = (
                        head_attn[query_start:query_end, ds:de].sum()
                    )

                # Rank of the gold tool (1-indexed, lower is better)
                gold_score = doc_scores_head[gold_tool_id]
                gold_rank  = (doc_scores_head > gold_score).sum().item() + 1

                # Accumulate MRR contribution
                head_scores[layer_idx, head_idx] += 1.0 / gold_rank

        if target_device.type == "cuda":
            del attentions, inputs
            torch.cuda.empty_cache()
            gc.collect()

    # ---------------------------------------------------------------
    # Select the top max_heads heads by accumulated MRR score
    # ---------------------------------------------------------------
    flat_scores   = head_scores.view(-1)                           # [L * H]
    top_k_indices = torch.topk(flat_scores, max_heads).indices     # [max_heads]

    selected_heads = [
        (idx.item() // num_heads, idx.item() % num_heads)
        for idx in top_k_indices
    ]

    # example expected format:
    # [(layer1, head3), (layer5, head10), ...]
    assert len(selected_heads) == max_heads
    return selected_heads