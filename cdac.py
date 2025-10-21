from data_process import *
from config import *
from model import *
from losses import *
from utils import *


import torch
import numpy as np
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
import torch.nn.functional as F
import logging
from tqdm import tqdm, trange


class CdacManager:

    def __init__(self, args, data_processor, logger_name='Main Training'):
        self.logger = logging.getLogger(logger_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = Bert(model=args.model, feat_dim=args.feat_dim).to(self.device)

        # 加载PretrainBert 的 backbone 权重
        pretrain_file = os.path.join(args.output_dir, "best_pretrain_model.pt")
        if os.path.exists(pretrain_file):
            pretrain_ckpt = torch.load(pretrain_file, map_location=self.device)
            # 只拿 "backbone.xxx" 开头的键
            backbone_dict = {k.replace("backbone.", ""): v
                             for k, v in pretrain_ckpt.items()
                             if k.startswith("backbone.")}
            self.model.backbone.load_state_dict(backbone_dict, strict=True)
            print("✅ PretrainBert backbone loaded into Bert.")
        else:
            print("⚠️  pretrain weight not found, train from scratch.")        

        self.train_labeled_dataloader = data_processor.train_labeled_dataloader
        self.eval_known_dataloader = data_processor.eval_known_dataloader
        self.train_semi_dataloader = data_processor.train_semi_dataloader
        self.test_dataloader = data_processor.test_dataloader
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

    def eval(self, args, dataloader):

        self.model.eval()

        total_feats = torch.empty((0,args.feat_dim)).to(self.device)
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="Evaluating", leave=False):

            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.set_grad_enabled(False):
                feats = self.model(
                    input_ids=batch['input_ids'], 
                    attention_mask=batch['attention_mask'], 
                    labels=None)

            total_feats = torch.cat((total_feats, feats))
            total_labels = torch.cat((total_labels, batch['label']))

        return total_feats, total_labels

    def train(self, args, eps=1e-10):

        wait = 0
        best_model = copy.deepcopy(self.model)
        best_metrics = {
            'Epoch': 0,
            'ACC': 0,
            'ARI': 0,
            'NMI': 0
        }   

        for epoch in trange(int(args.num_train_epochs), desc="CDAC"):
            self.model.train()
            tr_sim_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            # 动态更新阈值
            eta = epoch * 0.009  # 自适应参数
            u = max(0.5, 0.95 - eta)  # 防止u过小
            l = min(0.9, 0.455 + eta * 0.1)  # 防止l过大            

            for step, batch in enumerate(tqdm(self.train_semi_dataloader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with torch.set_grad_enabled(True):
                    seq_emb = self.model(
                        input_ids=batch['input_ids'], 
                        attention_mask=batch['attention_mask'], 
                        labels=None, )

                    # sim: [bsz, bsz], seq_emb: [bsz, feat_dim]
                    sim = torch.matmul(seq_emb, seq_emb.transpose(0, -1)) 
                    batch_R = self.get_global_R(y_true=batch["label"], sim=sim, l=l, u=u)
                    
                    # 计算相似度损失
                    pos_mask = (batch_R == 1)
                    neg_mask = (batch_R == 0)
                    pos_entropy = -torch.log(torch.clamp(sim, eps, 1.0)) * pos_mask
                    neg_entropy = -torch.log(torch.clamp(1 - sim, eps, 1.0)) * neg_mask

                    sim_loss = pos_entropy.mean() + neg_entropy.mean() + u - l

                    self.optimizer.zero_grad()
                    sim_loss.backward()
                    # nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                   
                    tr_sim_loss += sim_loss.item()

                    nb_tr_examples += batch['input_ids'].size(0)
                    nb_tr_steps += 1

            sim_loss = tr_sim_loss / nb_tr_steps

            # 直接用测试集评估
            test_feats, test_y_true = self.eval(args, dataloader=self.test_dataloader)

            # 用测试集集查看候选三元组的数量
            sim_mat = self.get_sim_score(feats=test_feats)
            # sim_mat = test_feats @ test_feats.T
            global_R = self.get_global_R(y_true=test_y_true, sim=sim_mat, l=l, u=u)
            indices_pairs = self.get_uncert_pairs(global_R)
            self.logger.info(f"Epoch {epoch}: u={u:.3f}, l={l:.3f}, uncertain pairs={len(indices_pairs)}")
            
            # 查看聚类指标
            test_feats = test_feats.cpu().numpy()
            test_y_true = test_y_true.cpu().numpy()

            km = KMeans(n_clusters = args.num_labels).fit(test_feats)
            test_y_pred = km.labels_
        
            # 这里已经用了匈牙利对齐算法
            test_results = clustering_score(test_y_true, test_y_pred)
            plot_cm = True
            if plot_cm:
                ind, _ = hungray_alignment(test_y_true, test_y_pred)
                map_ = {i[0]:i[1] for i in ind}
                test_y_pred = np.array([map_[idx] for idx in test_y_pred])

                cm = confusion_matrix(test_y_true,test_y_pred)               
            
            self.logger.info("***** Test: Confusion Matrix *****")
            self.logger.info("%s", str(cm))

            self.logger.info("***** Test results *****")
            for key in sorted(test_results.keys()):
                self.logger.info("  %s = %s", key, str(test_results[key]))

            if test_results['ACC'] + test_results['ARI'] + test_results['NMI'] > \
                best_metrics['ACC'] + best_metrics['ARI'] + best_metrics['NMI']:
                best_metrics['Epoch'] = epoch
                best_metrics['ACC'] = test_results['ACC']
                best_metrics['ARI'] = test_results['ARI']
                best_metrics['NMI'] = test_results['NMI']
                best_model = copy.deepcopy(self.model)
                wait = 0
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break                

        self.model = best_model
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, "best_cdac_model.pt")
        torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Best cdac model saved to {save_path}")



    def get_sim_score(self, feats):
        # feats 是 tensor
        # sim: [num_samples, num_samples]
        sim = torch.matmul(feats, feats.T)
        # 查看sim数据分布情况 - 使用 PyTorch 操作
        # 1. 只取下三角（去掉对角线）
        mask = torch.tril(torch.ones_like(sim), diagonal=-1).bool()        
        # 2. 把下三角相似度拉成一维
        flat = sim[mask]
        # 3. 使用 PyTorch 的 histc 进行统计
        bins = 10
        hist = torch.histc(flat, bins=bins, min=0, max=1)       
        # 4. 把计数转换为字符串
        info = ' '.join([f'{int(count)}' for count in hist])
        self.logger.info("sim distrib: %s", info)
        return sim

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
        global_R[other_mask & (sim >= u)] = torch.tensor(1.0).to(self.device)
        global_R[other_mask & (sim <= l)] = torch.tensor(0.0).to(self.device)
        return global_R
    
    def get_uncert_pairs(self, global_R):
        # get uncertainty pair
        # 使用顺序采样器对所有样本操作，这里使用的是全局索引
        uncert_mask = (global_R == -1)
        # 只保留下三角（去重+去掉对角线）
        # 这里需要注意，因为调用LLM有相同样本对但由于顺序不一致导致结果不一致的可能。
        # 但如果选择只保留下三角，那直接不会出现这个问题。
        mask = torch.tril(uncert_mask, diagonal=-1)
        row, col = torch.where(mask)

        # List[(i, j)]
        indices_pairs = torch.stack([row, col], dim=1).tolist()

        return indices_pairs


if __name__ == "__main__":

    args = init_model()

    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.output_dir):
        raise RuntimeError(f"Failed to create output directory: {args.output_dir}")
    
    log_path = os.path.join(args.output_dir, "cdac_train.log")

    logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()   # 控制台
    ]
    )

    data_processor = PrepareData(args)
    manager = CdacManager(args, data_processor, logger_name='Discovery')

    manager.train(args)

        