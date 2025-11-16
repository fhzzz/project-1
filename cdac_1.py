"""
- 这个基本和CDAC原论文第一阶段是一样的，模型直接返回损失，不过增加了早停机制以及保存评估结果最好的模型
- 和 cdac.py 最大的区别在于模型的设置不同
- 随机采样器，加载预训练模型，但是首轮效果跟没加载一样，不知道为什么。strict=True/False结果差别不大
- 应该是没加载成功预训练权重。
- 另外，评估指标的计算方法有一点小改正

- 模型保存在 results/clinc/best_cdac_model.pt, cdac_train_1109.log
"""

from data_process import *
from config import *
from model import *
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

        self.model = BertForConstrainClustering(model=args.model, num_labels=args.num_labels).to(self.device)

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


        self.train_labeled_dataloader = data_processor.train_labeled_dataloader
        self.eval_known_dataloader = data_processor.eval_known_dataloader
        self.test_dataloader = data_processor.test_dataloader        
        self.train_semi_samples = data_processor.train_semi_samples

        # DataLoader: 随机采样器 v.s. 顺序采样器
        # train_semi_sampler = RandomSampler(self.train_semi_samples)
        train_semi_sampler = SequentialSampler(self.train_semi_samples)
        self.train_semi_dataloader = DataLoader(dataset=self.train_semi_samples, batch_size=args.train_batch_size, 
                                                sampler=train_semi_sampler)   
        
        # self.index_to_text = data_processor.index_to_text
        steps = len(self.train_semi_dataloader) * args.num_train_epochs
        self.optimizer, self.scheduler = self.get_optimizer(args, steps)

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

    def evaluation(self, args, dataloader, plot_cm=True):

        self.model.eval()

        total_logits = torch.empty((0,args.num_labels)).to(self.device)
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="Evaluating", leave=False):

            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.set_grad_enabled(False):
                logits = self.model(
                    input_ids=batch['input_ids'], 
                    token_type_ids=batch['token_type_ids'],
                    attention_mask=batch['attention_mask'], 
                    mode=None)

            total_logits = torch.cat((total_logits, logits))
            total_labels = torch.cat((total_labels, batch['label']))

        total_logits = total_logits.cpu().numpy()
        total_labels = total_labels.cpu().numpy()
        total_preds = np.argmax(total_logits, 1)

        cluster_results = clustering_score(total_labels, total_preds)

        # confusion matrix
        if plot_cm:
            ind, _ = hungray_alignment(total_labels, total_preds)
            map_ = {i[0]:i[1] for i in ind}
            y_pred = np.array([map_[idx] for idx in total_preds])

            cm = confusion_matrix(total_labels, y_pred)
        return cluster_results, cm
              

    def train(self, args, eps=1e-10):

        u = 0.95
        l = 0.455
        eta = 0
        wait = 0
        best_metrics = {'ACC': 0, 'ARI': 0, 'NMI': 0, 'Epoch': 0}   

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for batch in tqdm(self.train_labeled_dataloader, desc="Iteration (labeled)"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                loss = self.model(input_ids=batch['input_ids'], 
                                  token_type_ids=batch['token_type_ids'], 
                                  attention_mask=batch['attention_mask'], 
                                  u_threshold=u, l_threshold=l, labels=batch['label'], 
                                  mode='train', semi=False)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                
                tr_loss += loss.item()
                nb_tr_examples += batch['input_ids'].size(0)
                nb_tr_steps += 1                

            train_labeled_loss = tr_loss / nb_tr_steps
            
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for batch in tqdm(self.train_semi_dataloader, desc="Iteration (all train)"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                loss = self.model(input_ids=batch['input_ids'], 
                                  token_type_ids=batch['token_type_ids'], 
                                  attention_mask=batch['attention_mask'], 
                                  u_threshold=u, l_threshold=l, labels=batch['label'], 
                                  mode='train', semi=True)
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                tr_loss += loss.item()
                nb_tr_examples += batch['input_ids'].size(0)
                nb_tr_steps += 1

            train_loss = tr_loss / nb_tr_steps
            # Evaluation
            cluster_results, cm = self.evaluation(args, self.test_dataloader, plot_cm=True)
            
            self.logger.info(f"*****Epoch {epoch}: Training  Results *****")
            self.logger.info(f"loss={train_loss}, (u, l) = ({round(u, 4)},{round(l, 4)})")

            eta += 1.1 * 0.009
            u = 0.95 - eta
            l = 0.455 + eta*0.1
            if u < l:
                break

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

            self.logger.info(f"***** Epoch {epoch}: Test results *****")
            result_line = "  " + "  ".join([f"{key} = {cluster_results[key]}" for key in cluster_results.keys()])
            best_result = "  " + "  ".join([f"{key} = {best_metrics[key]}" for key in best_metrics.keys()])
            self.logger.info(result_line)   
            self.logger.info(best_result)
            self.logger.info(f"当前最佳 epoch: {best_metrics["Epoch"]}, wait={wait}")
            self.logger.info("%s", str(cm))                    

        self.model = best_model                                                            

        os.makedirs(os.path.join(args.output_dir, args.dataset), exist_ok=True)
        save_path = os.path.join(args.output_dir, args.dataset, "best_cdac_model.pt")
        torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Best cdac model saved to {save_path}")


if __name__ == "__main__":

    args = init_model()

    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.output_dir):
        raise RuntimeError(f"Failed to create output directory: {args.output_dir}")
    
    log_path = os.path.join(args.output_dir, args.dataset, "main_cdac_1109.log")

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