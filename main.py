from data_process import *
from config import *
from model import *
from losses import *


import torch
import numpy as np
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
import torch.nn.functional as F
import logging
from tqdm import tqdm, trange

from openai import OpenAI
import tiktoken


class MainManager:

    def __init__(self, args, data_processor, logger_name='Main Training'):
        self.logger = logging.getLogger(logger_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = Bert(mode=args.model, ).to(self.device)

        data_processor = PrepareData()
        self.train_labeled_dataloader = data_processor.train_labeled_dataloader
        self.eval_known_dataloader = data_processor.eval_known_dataloader
        self.train_semi_dataloader = data_processor.train_semi_dataloader
        self.index_to_text = data_processor.index_to_text
        steps = len(self.train_labeled_dataloader) * args.num_train_epochs
        self.optimizer, self.scheduler = self.get_optimizer(args, steps)
        
        self.triplet_loss = 1
        self.centroids = None


    def get_optimizer(self, args, steps):
        num_warmup_steps = int(args.warmup_proportion * steps)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
                'weight_decay': 0.0
            }
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr_pre)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps= steps)
        return optimizer, scheduler

    def eval(self, args, dataloader, get_feats=False):

        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_preds = torch.empty(0, dtype=torch.long).to(self.device)

        total_feats = torch.empty((0, self.model.config.hidden_size)).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)

        for batch in tqdm(dataloader, desc="Iteration", leave=False):

            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.set_grad_enabled(False):
                sent_embed, logits = self.model(
                    input_ids=batch['input_ids'], 
                    attention_mask=batch['attention_mask'], 
                    labels=None, 
                    mode=None,
                )

                total_feats = torch.cat(total_feats, sent_embed)
                total_labels = torch.cat(total_labels, batch['label'])
                total_logits = torch.cat(total_logits, logits)
        
        if get_feats:
            feats = total_feats.cpu().numpy()
            y_true = total_labels.cpu().numpy()
            return feats, y_true
        else:
            total_probs = F.softmax(total_logits.detach(), dim=1)
            _, total_preds = total_probs.max(dim=1)

            y_pred = total_preds.cpu().numpy()
            y_true = total_labels.cpu().numpy()
            return y_true, y_pred

    def train(self, args):

        best_eval_score = 0
        wait = 0
        best_model = None

        # # 预训练阶段（如果尚未进行）
        # if not hasattr(self, 'pretrain_complete'):
        #     self.logger.info("开始预训练阶段...")
        #     self.pretrain_manager.train(args)
        #     self.pretrain_complete = True       

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            # 

            for step, batch in enumerate(tqdm(self.train_semi_dataloader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with torch.set_grad_enabled(True):
                    pass
                    

    def get_sim_score(self, feats):
        # feats: [num_samples, feat_dim]
        # sim: [num_samples, num_samples]
        feats = F.normalize(feats, p=2, dim=1)
        sim = torch.matmul(feats, feats.t())
        return sim

    def get_label_R(self, labels):
        # labels: [bsz,] or [num+label_samples]??
        label_R = labels.unsqueeze(0) == labels.unsqueeze(1)
        label_R = label_R.float()
        return label_R

    def update_threshholds(sim_score, ):
        pass


    def get_global_R(self, y_true, sim, l, u):
        # 初始化为-1
        global_R = torch.full_like(sim, -1.0)

        # 0) label sample and label sample
        mask_label = (y_true != -1)
        both_label_mask = mask_label.unsqueeze(0) & mask_label.unsqueeze(1)
        label_R = (y_true.unsqueeze(0) == y_true.unsqueeze(1)).float()
        global_R[both_label_mask] = label_R[both_label_mask]
        
        # 1) 其余所有情况: 至少一个样本没有标签
        other_mask = ~both_label_mask
        global_R[other_mask & (sim >= u)] = torch.tensor(1.0)
        global_R[other_mask & (sim <= l)] = torch.tensor(0.0)
        return global_R
    
    def get_uncert_pairs(self, global_R):
        # get uncertainty pair
        # 使用顺序采样器对所有样本操作，这里使用的是全局索引
        uncert_mask = (global_R == -1)
        # 只保留下三角（去重+去掉对角线）
        # 这里需要注意，因为调用LLM有相同样本对但由于顺序不一致导致结果不一致的可能。
        # 但如果选择只保留下三角，那直接不会出现这个问题。
        mask = torch.tril(uncert_mask, diagnonal=-1)
        row, col = torch.where(mask)

        # List[(i, j)]
        indices_pairs = torch.stack([row, col], dim=1).tolist()

        return indices_pairs

    def get_text_pairs(self, indices_pairs):
        """
        根据索引对获取对应的文本对
        
        参数:
            indices_pairs: 索引对列表 [(i, j), ...]
            
        返回:
            text_pairs: 文本对列表 [(text_i, text_j), ...]
        """
        text_pairs = []
        for i, j in indices_pairs:
            text_i = self.index_to_text["train"][i]
            text_j = self.index_to_text["train"][j]
            text_pairs.append((text_i, text_j))
        return text_pairs

    def llm_labeling(self, args, epoch, model, l, u):

        # 创建结果保存文件
        output_file = os.path.join(args.result_dir, "llm_outputs", \
            f"llm_annotated_output_{args.seed}_{args.known_class_ratio}_{epoch}.json")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        if os.path.exists(output_file):
            self.logger.info(f'加载缓存的LLM标注结果: {outout_file}')
            with open(output_file, 'r') as f:
                return json.load(f)
        
        self.logger.info(f'Starting LLM labeling, the number of samples is {len(text_pairs)}...')

        # # 0. 先按照整体分布聚类
        # feats, y_true = self.eval(args, dataloader=self.train_semi_dataloader, get_feats=True)
        # km = KMeans(n_clusters=self.num_classes).fit(feats)
        # cluster_centroids, y_pred = km.cluster_centers_, km.labels_
        # cluster_centroids, y_pred = self.alignment(self.centroids, cluster_centroids, y_pred)
        # self.centroids = cluster_centroids

        # # 应用匈牙利算法将预测结果映射到真实标签
        # # y_pred_map: 每个具体样本预测标签对应映射后的标签，
        # # cluster_map: 每个聚类中心对应映射后的标签
        # y_pred_map, cluster_map, cluster_map_opp = self.get_hungray_aligment(y_pred, y_true)

        # 1. get feats and labels
        # train_semi_dataloader是顺序采样器
        feats, y_true = self.eval(args, dataloader=self.train_semi_dataloader, get_feats=True)
        
        # 2. compute sim_score
        sim_score = self.get_sim_score(feats)

        # 3. get global matrix R and uncertainty pairs
        global_R = self.get_global_R(y_true=y_true, sim=sim_score, l=l, u=u)
        indices_pairs = self.get_uncert_pair(global_R=global_R)
        text_pairs = self.get_text_pairs(indices_pairs)

        # self.logger.info(f"[LLM] 不确定性样本对数量 = {len(text_pairs)}")

        # llm prompt
        # 初始化OpenAI客户端
        client = OpenAI(api_key=args.openai_api_key)

        SYS_PROMPT = (
            "You are a semantic equivalence judge. "
            "Given two sentences, determine whether they express the same meaning / intent."
        )

        USER_TEMPLATE = (
            "Sentence A: {sent_A}\n"
            "Sentence B: {sent_B}\n"
            "Do they express the same meaning? Please answer only \"Yes\" or \"No\" and then give a confidence score "
            "between 0.0 and 1.0 (e.g., \"Yes 0.92\"). Avoid any other words."
        )

        # 批量调用 + 容错
        # results = []
        llm_generated_outputs = {"pair_index": [], "llm_pred": [], "conf": []}
        for idx, (t1, t2) in tqdm(enumerate(text_pairs), total=len(text_pairs), desc="LLM"):
            prompt = USER_TEMPLATE.format(sent_A=t1, sent_B=t2)
            messages = [{"role": "system", "content": SYS_PROMPT}, 
            {"role": "user", "content": prompt}]
            tokens = ENC.encode(SYS_PROMPT + prompt)

            for attempt in range(args.max_retry):
                try:
                    response = client.chat.completions.create(
                        model=args.model_name, 
                        messages=messages, 
                        temperature=args.temperature, 
                        timeout=30, 
                    )

                    raw: str = response.choices[0].message.content.strip()
                    parts = raw.split()
                    if len(parts) != 2:
                        raise ValueError("格式不对")
                    yn, conf = parts[0].lower(), float(parts[1])
                    if len(parts) != 2:
                        raise ValueError("解析失败")
                    pred = 1 if yn == "yes" else 0
                    break
                except Exception as e:
                    self.logger,Warning(f"[LLM] 第{attempt+1}次重试失败：{e}")
                    time.sleep(random.uniform(1, 3))
                    pred, conf = -1, 0.0
            else:
                self.logger.error(f"[LLM] 最终失败，跳过该对")
                pred, conf = -1, 0.0
            
            i, j = indices_pairs[idx]
            llm_generated_outputs["pair_index"].append([i, j])
            llm_generated_outputs["llm_pred"].append(pred)
            llm_generated_outputs["conf"].append(conf)
        
        self.logger.info(f"[LLM] 标注完成，已写入{output_file}")
        return llm_generated_outputs

    # def seva_llm_results(self, args, results, ):
    #     filename = f"llm_labeling_results_{args.seed}_{args.known_class_ratio}.json"
    #     filepath = os.path.join(args.result_dir, filename)

    #     os.makedirs(os.path.dirname(filepath), exist_ok=True)
    #     with open(filepath, 'w') as f:
    #         json.dump(results, f, indent=4, ensure_ascii=False)

    #     self.logger.info(f"LLM结果已保存到: {filepath}")
    
    # def _parse_llm_response(self, response):
    #     if not response or "Error" in response:
    #         return -1, 0.0

    #     try:
    #         cleaned = response.strip().lower()
    #         if '\n' in cleaned:
    #             cleaned = 

    def update_R(self, global_R, llm_outputs, sim, y_true, conf_thresh):
        N = global_R.size(0)
        new_R = global_R.clone()

        for (i, j), pred, conf in zip(
            llm_outputs["pair_index"], 
            llm_outputs["llm_pred"], 
            llm_outputs["conf"]
        ):
            if conf >= conf_thresh and global_R[i, j] == -1:
                new_R[i, j] = new_R[j, i] =float(pred)
        return new_R




        





        

        






