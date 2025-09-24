from data_process import *
from config import *
from model import *
from losses import *

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typeing import List, Tuple, Dict
import logging
from tqdm import tqdm


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
                total_logits = torch.cat(total_logtis, logits)
        
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
                    







