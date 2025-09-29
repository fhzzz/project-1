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


class MainManager:

    def __init__(self, args, data_processor, logger_name='Main Training'):
        self.logger = logging.getLogger(logger_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = Bert(mode=args.model, ).to(self.device)

        data_processor = PrepareData()
        self.train_labeled_dataloader = data_processor.train_labeled_dataloader
        self.eval_known_dataloader = data_processor.eval_known_dataloader
        self.train_semi_dataloader = data_processor.train_semi_dataloader

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

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(self.train_semi_dataloader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with torch.set_grad_enabled(True):
                    pass
                    

    def get_sim_score(self, feats):
        # feats: []
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
    
    def get_uncert_pair(self, global_R):
        # 3) get uncertainty pair
        # 使用顺序采样器对所有样本操作，这里使用的是全局索引
        uncert_mask = (global_R == -1)
        # 只保留下三角（去重+去掉对角线）
        # 这里需要注意，因为调用LLM有相同样本对但由于顺序不一致导致结果不一致的可能。
        # 但如果选择只保留下三角，那直接不会出现这个问题。
        mask = torch.tril(uncert_mask, diagnonal=-1)
        row, col = torch.where(mask)

        # List[(i, j)]
        uncert_ij = torch.stack([row, col], dim=1).tolist()

        return uncert_ij

    def llm_labeling(self, args, epoch, model, l, u):

        # 创建结果保存文件

        self.logger.info('Start LLM labeling ...')

        # 0. 先按照整体分布聚类
        feats, y_true = self.eval(args, dataloader=self.train_semi_dataloader, get_feats=True)
        km = KMeans(n_clusters=self.num_classes).fit(feats)
        cluster_centroids, y_pred = km.cluster_centers_, km.labels_
        cluster_centroids, y_pred = self.alignment(self.centroids, cluster_centroids, y_pred)
        self.centroids = cluster_centroids

        # 应用匈牙利算法将预测结果映射到真实标签
        # y_pred_map: 每个具体样本预测标签对应映射后的标签，
        # cluster_map: 每个聚类中心对应映射后的标签
        y_pred_map, cluster_map, cluster_map_opp = self.get_hungray_aligment(y_pred, y_true)

        # 1. get feats and labels
        # train_semi_dataloader是顺序采样器
        feats, y_true = self.eval(args, dataloader=self.train_semi_dataloader, get_feats=True)
        
        # 2. compute sim_score
        sim_score = self.get_sim_score(feats)

        # 3. get global matrix R and uncertainty pairs
        global_R = self.get_global_R(y_true=y_true, sim=sim_score, l=l, u=u)
        uncert_ij = self.get_uncert_pair(global_R=global_R)

        # LLM outputs
        llm_generated_outputs = {
            "pair_index": [], 
            "llm_pred": [], 
        }

        price_usage = 0
        for _, (i, j) in tqdm(enumerate()):
            pass




        

        






