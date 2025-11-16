"""
- 目的在于探究、解决评估指标下降的问题
- 修改评估方式
- 阈值设置为按照数据分布进行

"""

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
        self.logger.info(f"device: {self.device}")


        self.num_labels = data_processor.num_labels
        self.train_labeled_dataloader = data_processor.train_labeled_dataloader
        self.eval_known_dataloader = data_processor.eval_known_dataloader
        self.test_dataloader = data_processor.test_dataloader
        self.train_semi_samples = data_processor.train_semi_samples

        # DataLoader: 随机采样器 v.s. 顺序采样器
        train_semi_sampler = RandomSampler(self.train_semi_samples)
        # train_semi_sampler = SequentialSampler(self.train_semi_samples)
        self.train_semi_dataloader = DataLoader(dataset=self.train_semi_samples, batch_size=args.train_batch_size, 
                                                sampler=train_semi_sampler)   

        self.model = Bert(model=args.model, feat_dim=args.feat_dim, num_labels=self.num_labels).to(self.device)

        # 加载PretrainBert 的 backbone 权重
        pretrain_file = os.path.join(args.output_dir, args.dataset, "best_pretrain_model.pt")
        if os.path.exists(pretrain_file):
            pretrain_ckpt = torch.load(pretrain_file, map_location=self.device)
            # 只拿 "backbone.xxx" 开头的键
            backbone_dict = {k.replace("backbone.", ""): v
                             for k, v in pretrain_ckpt.items()
                             if k.startswith("backbone.")}
            self.model.backbone.load_state_dict(backbone_dict, strict=True)
            self.logger.info("✅ PretrainBert backbone loaded into Bert.")
        else:
            self.logger.info("⚠️  pretrain weight not found, train from scratch.")


        # self.index_to_text = data_processor.index_to_text
        steps = len(self.train_semi_dataloader) * args.num_train_epochs
        self.optimizer, self.scheduler = self.get_optimizer(args, steps)
        self.centroids = None

        # 记录超参数信息
        self.logger.info(f"{args.dataset}, KCL={args.known_cls_ratio}, train_batch_size={args.train_batch_size}")


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
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps= steps)
        return optimizer, scheduler

    def get_features(self, args, dataloader):

        self.model.eval()

        total_feats = torch.empty((0,args.feat_dim)).to(self.device)
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="Evaluating", leave=False):

            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.set_grad_enabled(False):
                feats = self.model(
                    input_ids=batch['input_ids'], 
                    token_type_ids=batch['token_type_ids'], 
                    attention_mask=batch['attention_mask'], 
                    labels=None, mode="feat_ext")

            total_feats = torch.cat((total_feats, feats))
            total_labels = torch.cat((total_labels, batch['label']))

        return total_feats, total_labels

    def evaluation(self, args, plot_cm=True):
        """final clustering evaluation on test set"""
        # get features
        feats, labels = self.get_features(args, self.test_dataloader)
        feats = feats.cpu().numpy()
        # k-means clustering
        km = KMeans(n_clusters = self.num_labels, random_state=args.seed).fit(feats)
        y_pred = km.labels_
        y_true = labels.cpu().numpy()

        cluster_results = clustering_score(y_true, y_pred)

        if plot_cm:
            ind, _ = hungray_alignment(y_true, y_pred)
            map_ = {i[0]:i[1] for i in ind}
            y_pred = np.array([map_[idx] for idx in y_pred])

            cm = confusion_matrix(y_true,y_pred)

        return cluster_results, cm 
    
    def eval(self, args, dataloader):
        """和evaluation的区别在于使用分类指标还是聚类评估"""
        self.model.eval()

        total_logits = torch.empty((0,self.num_labels)).to(self.device)
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="Evaluating", leave=False):

            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.set_grad_enabled(False):
                logits = self.model(
                    input_ids=batch['input_ids'], 
                    token_type_ids=batch['token_type_ids'], 
                    attention_mask=batch['attention_mask'], 
                    labels=None, mode=None)

            total_logits = torch.cat((total_logits, logits))
            total_labels = torch.cat((total_labels, batch['label']))

        total_labels = total_labels.cpu().numpy()
        total_logits = total_logits.cpu().numpy()
        total_preds = np.argmax(total_logits, 1)
        # total_probs = F.softmax(total_logits.detach(), dim=1)
        # _, total_preds = total_probs.max(dim=1)
        assert total_logits.shape[1] == self.num_labels

        return total_preds, total_labels        
        

    def train(self, args, eps=1e-10):

        wait = 0
        u = 0.95
        l = 0.455
        eta = 0        
        best_score = 0  
        best_metrics = {'ACC': 0, 'ARI': 0, 'NMI': 0, 'Epoch': 0}        

        for epoch in trange(int(args.num_train_epochs), desc="CDAC"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            num_pairs = 0

            for step, batch in enumerate(tqdm(self.train_semi_dataloader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with torch.set_grad_enabled(True):
                    seq_emb = self.model(
                        input_ids=batch['input_ids'],
                        token_type_ids=batch['token_type_ids'],  
                        attention_mask=batch['attention_mask'], 
                        labels=None)

                    # sim: [bsz, bsz], seq_emb: [bsz, feat_dim]
                    sim_mat = torch.matmul(seq_emb, seq_emb.transpose(0, -1))
                    # sim_mat = (sim_mat + 1) / 2  # 大NO特NO
                    # sim_mat = sim_mat / args.temperature

                    # if step == 0:  # 每个epoch的第一个batch重新计算
                    #     u, l = self.update_u_l(sim_mat)
                    if step == 0:
                        self.logger.info("***** Epoch: %s: Similarities Distrib*****", str(epoch))
                    if step % 100 == 0:
                        self.get_sim_distrib(sim_mat)                    
                    batch_R = self.get_batch_R(y_true=batch['label'], sim=sim_mat, l=l, u=u)                                      
                    uncert_pairs = self.get_uncert_pairs(batch_R)
                    num_pairs += len(uncert_pairs)
                    
                    # 计算相似度损失
                    pos_mask = (batch_R == 1)
                    neg_mask = (batch_R == 0)
                    pos_entropy = -torch.log(torch.clamp(sim_mat, eps, 1.0)) * pos_mask
                    neg_entropy = -torch.log(torch.clamp(1 - sim_mat, eps, 1.0)) * neg_mask

                    loss = pos_entropy.mean() + neg_entropy.mean() + u - l
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()

                    tr_loss += loss.item()
                    nb_tr_examples += batch['input_ids'].size(0)
                    nb_tr_steps += 1

                sim_loss = tr_loss / nb_tr_steps        
            
            # 评估
            y_pred, y_true = self.eval(args, self.test_dataloader)
            eval_score = round(accuracy_score(y_true, y_pred) * 100, 2)


            if eval_score > best_score:
                best_score = eval_score
                best_model = copy.deepcopy(self.model)
                wait = 0
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break      

            self.logger.info("***** Epoch: %s: Train Results and eval Results*****", str(epoch))
            self.logger.info(f"loss={sim_loss:.4f}, num_uncert_pairs={num_pairs}, (u, l) = ({round(u, 4)},{round(l, 4)})")                 
            self.logger.info(f"eval_score={eval_score}, best_score={best_score}, wait={wait}") 

            # 聚类指标评估            
            test_results, cm = self.evaluation(args, self.test_dataloader)
            if test_results['ACC'] + test_results['ARI'] + test_results['NMI'] > \
                best_metrics['ACC'] + best_metrics['ARI'] + best_metrics['NMI']:
                best_metrics['Epoch'] = epoch
                best_metrics['ACC'] = test_results['ACC']
                best_metrics['ARI'] = test_results['ARI']
                best_metrics['NMI'] = test_results['NMI']
            self.logger.info("***** Epoch: %s: Test Cluster Results*****", str(epoch))                       
            result_line = "  " + "  ".join([f"{key} = {test_results[key]}" for key in test_results.keys()])
            best_result = "  " + "  ".join([f"{key} = {best_metrics[key]}" for key in best_metrics.keys()])
            self.logger.info(result_line)   
            self.logger.info(best_result)         
            self.logger.info("%s", str(cm))            

            eta += 1.1 * 0.009
            u = 0.95 - eta
            l = 0.455 + eta*0.1
            if u < l:
                break                     

        self.model = best_model
        os.makedirs(os.path.join(args.output_dir, args.dataset), exist_ok=True)
        save_path = os.path.join(args.output_dir, args.dataset, "best_model.pt")
        torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Best cdac model saved to {save_path}")


    def get_batch_R(self, y_true, sim, l, u):
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

    def get_sim_distrib(self, sim):
        """统计分布情况"""
        mask = torch.tril(torch.ones_like(sim), diagonal=-1).bool()
        flat = sim[mask]
        bins = 10
        hist = torch.histc(flat, bins=bins, min=-1, max=1)
        info = ' '.join([f'{int(count)}' for count in hist])
        self.logger.info("sim distrib: %s", info)


    def update_u_l(self, similarities):
        """基于数据分布自动设置初始阈值"""
        u_threshold = torch.quantile(similarities, 0.95)  # 95%分位数
        l_threshold = torch.quantile(similarities, 0.05)  # 5%分位数
        # 确保阈值合理性
        u_threshold = max(0.7, min(0.98, u_threshold))
        l_threshold = max(0.3, min(0.6, l_threshold))
        
        # 确保u > l
        if u_threshold <= l_threshold:
            u_threshold = l_threshold + 0.1
            
        return u_threshold, l_threshold        



if __name__ == "__main__":

    args = init_model()

    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.output_dir):
        raise RuntimeError(f"Failed to create output directory: {args.output_dir}")
    
    log_path = os.path.join(args.output_dir, args.dataset, "cdac_3_1116.log")

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

        