from data_process import *
from config import *
from model import *
from losses import *


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

# from openai import OpenAI
# import tiktoken


class Main2Manager:

    def __init__(self, args, data_processor, logger_name='Main Training'):
        self.logger = logging.getLogger(logger_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = Bert(model=args.model, feat_dim=args.feat_dim).to(self.device)

        # 加载PretrainBert 的 backbone 权重
        pretrain_file = os.path.join(args.output_dir, args.dataset, "best_pretrain_model.pt")
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
        self.test_dataloader = data_processor.test_dataloader        
        self.train_semi_samples = data_processor.train_semi_samples
        # DataLoader: 顺序采样器
        train_semi_sampler = SequentialSampler(self.train_semi_samples)
        self.train_semi_dataloader = DataLoader(dataset=self.train_semi_samples, batch_size=args.train_batch_size, 
                                                sampler=train_semi_sampler) 

        self.index_to_text = data_processor.index_to_text

        steps = len(self.train_semi_dataloader) * args.num_train_epochs
        self.n_samples = len(self.train_semi_samples)

        self.optimizer, self.scheduler = self.get_optimizer(args, steps)
        self.tri_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        self.centroids = None
        self.num_labels = args.num_labels


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
                    attention_mask=batch['attention_mask'], 
                    labels=None)

            total_feats = torch.cat((total_feats, feats))
            total_labels = torch.cat((total_labels, batch['label']))

        return total_feats, total_labels


    def evaluation(self, args, plot_cm=True):
        """final clustering evaluation on test set"""
        # get features
        test_feats, labels = self.get_features(args, self.test_dataloader)
        test_feats = test_feats.cpu().numpy()
        # k-means clustering
        km = KMeans(n_clusters = args.num_labels, random_state=args.seed).fit(test_feats)
        y_pred = km.labels_
        y_true = labels.cpu().numpy()

        cluster_results = clustering_score(y_true, y_pred)

        if plot_cm:
            ind, _ = hungray_alignment(y_true, y_pred)
            map_ = {i[0]:i[1] for i in ind}
            y_pred = np.array([map_[idx] for idx in y_pred])

            cm = confusion_matrix(y_true,y_pred)

        return cluster_results, cm      


    def train(self, args, eps=1e-10):

        wait = 0
        best_metrics = {'ACC': 0, 'ARI': 0, 'NMI': 0, 'Epoch': 0}

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_sim_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            all_indices_pairs = []

            # 动态更新阈值
            eta = epoch * 0.009
            u = max(0.5, 0.95 - eta)
            l = min(0.9, 0.455 + eta * 0.1)             

            # stage-1: 按照batch计算相似度损失，同时收集难样本对
            for step, batch in enumerate(tqdm(self.train_semi_dataloader, desc="Phase 1", leave=False)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                bsz = batch['input_ids'].size(0)

                with torch.set_grad_enabled(True):
                    feats = self.model(input_ids=batch['input_ids'], 
                        token_type_ids=batch['token_type_ids'], 
                        attention_mask=batch['attention_mask'], 
                        labels=None)
                    
                    # sim: [bsz, bsz], seq_emb: [bsz, feat_dim]
                    sim = torch.matmul(feats, feats.transpose(0, -1)) 
                    if step % 100 == 0:
                        sim_distrib = self.get_sim_distrib(sim)

                    # 遗留问题：区间阈值到底应该怎么弄？感觉按照收紧不是很合理
                    # 如果使用表征学习的思想，那应该采用两阶段训练的思想
                    batch_R = self.get_global_R(y_true=batch["label"], sim=sim, l=l, u=u)
                    
                    # 计算相似度损失
                    pos_mask = (batch_R == 1)
                    neg_mask = (batch_R == 0)
                    pos_entropy = -torch.log(torch.clamp(sim, eps, 1.0)) * pos_mask
                    neg_entropy = -torch.log(torch.clamp(1 - sim, eps, 1.0)) * neg_mask

                    # 这里为什么要加 u-1 这一项？
                    sim_loss = pos_entropy.mean() + neg_entropy.mean() + u - l
                    # 损失函数是越多越好吗？要如何判断呢？我该用什么损失呢？
                    
                    indices_pairs = self.get_uncert_pairs(batch_R)
                    if indices_pairs:
                        # 将batch内索引转换为全局索引
                        global_indices_pairs = [
                            (i + step * bsz, j + step * bsz) 
                            for i, j in indices_pairs
                        ]
                        all_indices_pairs.extend(global_indices_pairs)

                    # # 反向传播相似度损失
                    # self.optimizer.zero_grad()
                    # sim_loss.backward()
                    # nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    # self.optimizer.step()
                    # self.scheduler.step()
                   
                    # tr_sim_loss += sim_loss.item()
                    # nb_tr_examples += bsz
                    # nb_tr_steps += 1

            # 第二阶段：使用收集到的难样本对构建三元组进行训练                   
            if all_indices_pairs:
                self.logger.info(f"Epoch {epoch}: collected {len(all_indices_pairs)} uncertain pairs")
                
                # 获取样本对的伪标签关系
                pair_relations = self.get_pesudo_relations_model(args, all_indices_pairs)
                
                # 构建三元组数据集
                triplet_dataset = self.update_dataset(all_indices_pairs, pair_relations)
                self.logger.info(f"len(triplet_dataset): {len(triplet_dataset)}")
                if len(triplet_dataset) > 0:
                    triplet_loader = DataLoader(
                        triplet_dataset, 
                        batch_size=args.train_batch_size,
                        shuffle=True
                    )
                    
                    # 使用三元组损失进行训练
                    tr_triplet_loss = 0
                    for triplet_batch in tqdm(triplet_loader, desc="Phase 2", leave=False):
                        triplet_batch = {k: v.to(self.device) for k, v in triplet_batch.items()}
                        
                        # 计算三元组损失
                        anchor_feats = self.model(
                            input_ids=triplet_batch['anchor_input_ids'],
                            attention_mask=triplet_batch['anchor_attention_mask'])

                        pos_feats = self.model(
                            input_ids=triplet_batch['positive_input_ids'],
                            attention_mask=triplet_batch['positive_attention_mask'])

                        neg_feats = self.model(
                            input_ids=triplet_batch['negative_input_ids'],
                            attention_mask=triplet_batch['negative_attention_mask'])
                        
                        triplet_loss = self.tri_loss(anchor_feats, pos_feats, neg_feats)
                        
                        self.optimizer.zero_grad()
                        triplet_loss.backward()
                        self.optimizer.step()
                        self.scheduler.step()
                        
                        tr_triplet_loss += triplet_loss.item()
                    
                    # self.logger.info(f"Epoch {epoch} - Sim Loss: {tr_sim_loss/nb_tr_steps:.4f}, Triplet Loss: {tr_triplet_loss/len(triplet_loader):.4f}")
                    self.logger.info(f"Epoch {epoch} -Triplet Loss: {tr_triplet_loss/len(triplet_loader):.4f}")
            # 评估
            cluster_results = self.evaluation(args, plot_cm=True)
            
            if cluster_results['ACC'] + cluster_results['ARI'] + cluster_results['NMI'] > \
                best_metrics['ACC'] + best_metrics['ARI'] + best_metrics['NMI']:
                best_metrics['Epoch'] = epoch
                best_metrics['ACC'] = cluster_results['ACC']
                best_metrics['ARI'] = cluster_results['ARI']
                best_metrics['NMI'] = cluster_results['NMI']
                best_model = copy.deepcopy(self.model)
                wait = 0
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break      

            self.logger.info("***** Test results *****")
            result_line = "  " + "  ".join([f"{key} = {cluster_results[key]}" for key in cluster_results.keys()])
            best_result = "  " + "  ".join([f"{key} = {best_metrics[key]}" for key in best_metrics.keys()])
            self.logger.info(result_line)   
            self.logger.info(best_result)
            self.logger.info(f"当前最佳 epoch: {best_metrics["Epoch"]}, wait={wait}")         

        self.model = best_model
        os.makedirs(os.path.join(args.output_dir, args.dataset), exist_ok=True)
        save_path = os.path.join(args.output_dir, args.dataset, "best_model.pt")
        torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Best cdac model saved to {save_path}")


    def get_pesudo_relations_model(self, args, indices_pairs):
        """
        参数：
            indices_pairs: 已经筛选好的样本对的索引，详见get_uncet_pairs方法
        return:
            想返回一个保存当前样本对索引信息，对应的由模型得到的伪标签关系，字典或者列表形式都可
        备注：
            我后面并没有写完整，思路并不一定正确，以你的为准
        """

        feats, y_true = self.get_features(args, self.train_semi_dataloader)
        feats = feats.cpu().numpy()
        y_true = y_true.cpu().numpy()
        km = KMeans(n_clusters=args.num_labels).fit(feats)
        cluster_centroids, y_pred = km.cluster_centers_, km.labels_
        # 匈牙利算法进行对齐
        cluster_centroids, y_pred = self.alignment(self.centroids, cluster_centroids, y_pred)
        self.centroids = cluster_centroids
        # 匈牙利算法将预测结果映射到真实标签：
        # y_pred_map: 每个具体样本预测标签对应映射后的标签，
        # cluster_map: 每个聚类中心对应映射后的标签
        y_pred_map, cluster_map, cluster_map_opp = self.get_hungray_alignment(y_pred, y_true)

        pair_relations = {}
        count_positive = 0
        for i, j in indices_pairs:
            if y_pred[i] == y_pred[j]:
                pair_relations[(i, j)] = 1
                count_positive += 1
            else:
                pair_relations[(i, j)] = 0

        total_pairs = len(indices_pairs)
        positive_ratio = count_positive / total_pairs if total_pairs > 0 else 0
        # self.logger.info(f"总样本对: {total_pairs}, 正例对: {count_positive}, 比例: {positive_ratio:.2f}")
        return pair_relations


    def update_dataset(self, indices_pairs, pair_relations):
        """构建三元组训练数据集"""
        triplets = []
        pos_count = sum(1 for rel in pair_relations.values() if rel == 1)  # 统计正例对数量
        neg_count = sum(1 for rel in pair_relations.values() if rel == 0)  # 统计负例对数量        
        self.logger.info(f"正例对数量: {pos_count}, 负例对数量: {neg_count}")

        # 遍历样本对和它们的关系
        for (anchor_idx, other_idx), relation in pair_relations.items():
            if relation == 1:  # 正例对
                # 在样本对中寻找负例
                neg_candidates = [idx for idx, rel in pair_relations.items() 
                                if idx[0] == anchor_idx and rel == 0]
                if neg_candidates:
                    neg_idx = random.choice(neg_candidates)[1]
                    triplets.append({
                        'anchor': self.train_semi_samples[anchor_idx],
                        'positive': self.train_semi_samples[other_idx],
                        'negative': self.train_semi_samples[neg_idx]
                    })
                neg_indices = np.random.choice(
                    [i for i in range(self.n_samples) if i != anchor_idx and i != other_idx],
                    size=args.k_neg,
                    replace=False
                )
                for neg_idx in neg_indices:
                    triplets.append({
                        'anchor': self.train_semi_samples[anchor_idx],
                        'positive': self.train_semi_samples[other_idx],
                        'negative': self.train_semi_samples[neg_idx]
                    })
            else:  # 负例对
                # 在训练集中寻找正例
                pos_candidates = [idx for idx, rel in pair_relations.items() 
                                if idx[0] == anchor_idx and rel == 1]
                if pos_candidates:
                    pos_idx = random.choice(pos_candidates)[1]
                    triplets.append({
                        'anchor': self.train_semi_samples[anchor_idx],
                        'positive': self.train_semi_samples[pos_idx],
                        'negative': self.train_semi_samples[other_idx]
                    })
        
        # 创建TripletDataset
        return TripletDataset(triplets)

            
    def alignment(self, old_centroids, new_centroids, cluster_labels):
        self.logger.info("***** Conducting Alignment *****")
        if old_centroids is not None:

            old_centroids = old_centroids
            new_centroids = new_centroids
            
            DistanceMatrix = np.linalg.norm(old_centroids[:,np.newaxis,:]-new_centroids[np.newaxis,:,:],axis=2) 
            row_ind, col_ind = linear_sum_assignment(DistanceMatrix)
            
            aligned_centroids = np.zeros_like(old_centroids)
            alignment_labels = list(col_ind)

            for i in range(self.num_labels):
                label = alignment_labels[i]
                aligned_centroids[i] = new_centroids[label]
            # 新label对应老label
            pseudo2label = {label:i for i,label in enumerate(alignment_labels)}
            pseudo_labels = np.array([pseudo2label[label] for label in cluster_labels])

        else:
            aligned_centroids = new_centroids    
            pseudo_labels = cluster_labels 

        self.logger.info("***** Update Pseudo Labels With Real Labels *****")
        
        return aligned_centroids, pseudo_labels


    def get_hungray_alignment(self, y_pred, y_true):
        num_test_samples = len(y_pred)
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D))
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
        y_pred_map = []
        cluster_map = [0]*len(ind)
        cluster_map_opp = [0]*len(ind)
        for i in range(num_test_samples):
            yp = y_pred[i]
            y_pred_map.append(ind[yp][1])
        y_pred_map = np.asarray(y_pred_map)

        for item in ind:
            cluster_map[item[0]] = item[1]
            cluster_map_opp[item[1]] = item[0]
        cluster_map = np.asarray(cluster_map)
        cluster_map_opp = np.asarray(cluster_map_opp)
        assert np.all(cluster_map[cluster_map_opp] == np.arange(len(ind)))
        return y_pred_map, cluster_map, cluster_map_opp


    def get_sim_distrib(self, sim):
        # sim = torch.clamp(sim, 0.0, 1.0)  # 限制在[0,1]区间

        # 统计分布情况
        mask = torch.tril(torch.ones_like(sim), diagonal=-1).bool()
        flat = sim[mask]
        
        bins = 10
        hist = torch.histc(flat, bins=bins, min=-1, max=1)
        info = ' '.join([f'{int(count)}' for count in hist])
        self.logger.info("sim distrib: %s", info)

    def get_label_R(self, labels):
        # labels: [bsz,] or [num+label_samples]??
        label_R = labels.unsqueeze(0) == labels.unsqueeze(1)
        label_R = label_R.float()
        return label_R

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
        mask = torch.tril(uncert_mask, diagonal=-1)
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
    

class TripletDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets
        
    def __len__(self):
        return len(self.triplets)
        
    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        
        return {
            'anchor_input_ids': triplet['anchor']['input_ids'],
            'anchor_attention_mask': triplet['anchor']['attention_mask'],
            'positive_input_ids': triplet['positive']['input_ids'],
            'positive_attention_mask': triplet['positive']['attention_mask'],
            'negative_input_ids': triplet['negative']['input_ids'],
            'negative_attention_mask': triplet['negative']['attention_mask']
        }


if __name__ == "__main__":

    args = init_model()

    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.output_dir):
        raise RuntimeError(f"Failed to create output directory: {args.output_dir}")
    
    log_path = os.path.join(args.output_dir, args.dataset, "train_model_1101.log")

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
    manager = Main2Manager(args, data_processor, logger_name='Discovery')
    manager.train(args)
        
