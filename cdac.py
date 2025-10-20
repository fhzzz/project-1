from data_process import *
from config import *
from model import *
from losses import *
from utils import *


import torch
import numpy as np
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
import torch.nn.functional as F
import logging
from tqdm import tqdm, trange

from openai import OpenAI
import tiktoken


class CdacManager:

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

        for batch in tqdm(dataloader, desc="Evaluating", leave=False):

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
            total_probs = F.softmax(total_logits.detach(), dim=1)
            _, total_preds = total_probs.max(dim=1)
            y_pred = total_preds.cpu().numpy()            
            return feats, y_true, y_pred
        else:
            total_probs = F.softmax(total_logits.detach(), dim=1)
            _, total_preds = total_probs.max(dim=1)

            y_pred = total_preds.cpu().numpy()
            y_true = total_labels.cpu().numpy()
            return y_true, y_pred

    def train(self, args, eps=1e-10):

        best_eval_score = 0
        wait = 0
        best_model = None    

        for epoch in trange(int(args.num_train_epochs), desc="CDAC"):
            self.model.train()
            tr_loss = 0
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
                        labels=None, 
                        mode='simple_forward')

                    # sim: [bsz, bsz], seq_emb: [bsz, feat_dim]
                    sim = self.get_sim_score(feats=seq_emb)  
                    batch_R = self.get_global_R(y_true=batch["labels"], sim=sim, l=l, u=u)
                    
                    # 计算相似度损失
                    pos_mask = (batch_R == 1)
                    neg_mask = (batch_R == 0)
                    pos_entropy = -torch.log(torch.clamp(sim, eps, 1.0)) * pos_mask
                    neg_entropy = -torch.log(torch.clamp(1 - sim, eps, 1.0)) * neg_mask

                    sim_loss = pos_entropy.mean() + neg_entropy.mean() + u - l

                    self.optimizer.zero_grad()
                    sim_loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                   
                    tr_sim_loss += sim_loss.item()

                    nb_tr_examples += batch['input_ids'].size(0)
                    nb_tr_steps += 1

            sim_loss = tr_sim_loss / nb_tr_steps

            # 直接用测试集评估
            test_feats, test_y_true, test_y_pred = self.eval(args, dataloader=self.test_dataloader, get_feats=True)
            test_score = round(accuracy_score(test_y_true, test_y_pred) * 100, 2)

            eval_results = {
                'Train_loss': round(loss, 6),
                'Train_ce_loss': round(ce_loss, 6),
                'Train_mlm_loss': round(mlm_loss, 6),
                'Eval_score': eval_score,
                'Best_score':best_eval_score,
                'Wait_epoch': wait, 
                'len(uncert_pairs)': len(indices_pairs)
            }

            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch))
            for key in eval_results.keys():
                self.logger.info("  %s = %s", key, str(eval_results[key]))

            # 用测试集集查看候选三元组的数量
            sim_mat = self.get_sim_score(feats=test_feats)
            global_R = self.get_global_R(sim_mat, u, l)
            indices_pairs = self.get_uncert_pairs(global_R)
            self.logger.info(f"候选三元组对的数量为: {len(indices_pairs)}")

            # 查看聚类指标
            km = KMeans(n_clusters = args.num_labels).fit(feats)
            y_pred = km.labels_
        
            test_results = clustering_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            
            self.logger.info
            self.logger.info("***** Test: Confusion Matrix *****")
            self.logger.info("%s", str(cm))
            self.logger.info("***** Test results *****")
            
            # 查看 u, l 的变化
            test_results["upper"] = u
            test_results["lower"] = l

            for key in sorted(test_results.keys()):
                self.logger.info("  %s = %s", key, str(test_results[key]))

            
            if eval_score > best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                best_eval_score = eval_score
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

    def test(self, args):
        self.logger.info('Testing cadc-km model...')
        feats, y_true = self.eval(args, dataloader=self.test_dataloader, get_feats=True)
        km = KMeans(n_clusters = args.num_labels).fit(feats)
        y_pred = km.labels_
    
        test_results = clustering_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        self.logger.info
        self.logger.info("***** Test: Confusion Matrix *****")
        self.logger.info("%s", str(cm))
        self.logger.info("***** Test results *****")
        
        for key in sorted(test_results.keys()):
            self.logger.info("  %s = %s", key, str(test_results[key]))

        test_results['y_true'] = y_true
        test_results['y_pred'] = y_pred
        return test_results


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

        