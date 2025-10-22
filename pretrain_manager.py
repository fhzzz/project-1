from data_process import *
from config import *
from model import *
from utils import *

from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score

import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, Dataset
import copy, logging

class PretrainManager:

    def __init__(self, args, data_processor, logger_name='Discovery'):
        
        self.logger = logging.getLogger(logger_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PretrainBert(model=args.model, num_labels=args.num_labels).to(self.device)
        
        self.train_labeled_datasets = data_processor.train_labeled_samples
        self.train_labeled_dataloader = data_processor.train_labeled_dataloader
        self.eval_known_datasets = data_processor.eval_known_samples
        self.eval_known_dataloader = data_processor.eval_known_dataloader
        self.train_mlm_dataloader = data_processor.train_mlm_dataloader
        # self.train_labeled_dataloader = DataLoader(dataset=self.train)

        steps = len(self.train_labeled_dataloader) * args.num_pretrain_epochs
        self.optimizer, self.scheduler = self.get_optimizer(args, steps)
        self.ce_loss = nn.CrossEntropyLoss()
    
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


    def eval(self, args, get_feats=False):

        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_preds = torch.empty(0, dtype=torch.long).to(self.device)

        total_feats = torch.empty((0, self.model.config.hidden_size)).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)

        for batch in tqdm(self.eval_known_dataloader, desc="Iteration", leave=False):

            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.set_grad_enabled(False):
                sent_embed, logits = self.model(
                    input_ids=batch['input_ids'], 
                    attention_mask=batch['attention_mask'], 
                    labels=None, 
                    mode=None, 
                )

                total_labels = torch.cat((total_labels, batch['label']))
                total_feats = torch.cat((total_feats, sent_embed))
                total_logits = torch.cat((total_logits, logits))
        
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

        for epoch in trange(int(args.num_pretrain_epochs), desc="Epoch"):
            
            self.model.train()
            tr_loss, tr_ce_loss, tr_mlm_loss = 0, 0, 0
            nb_tr_examples, nb_tr_steps = 0, 0
            semi_iter = iter(self.train_mlm_dataloader)

            for step, batch in enumerate(tqdm(self.train_labeled_dataloader)):
                
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # load semi batch
                try:
                    semi_batch = next(semi_iter)
                    semi_batch = {k: v.to(self.device) for k, v in semi_batch.items()}
                except StopIteration:
                    semi_iter = iter(self.train_mlm_dataloader)
                    semi_batch = next(semi_iter)
                    semi_batch = {k: v.to(self.device) for k, v in semi_batch.items()}

                # forward pass
                with torch.set_grad_enabled(True):
                    # cls
                    logits = self.model(
                        input_ids=batch['input_ids'], 
                        attention_mask=batch['attention_mask'], 
                        labels=None, 
                        mode="cls"
                    )
                    ce_loss = nn.CrossEntropyLoss()(logits, batch['label'])
                    # mlm
                    mlm_loss = self.model(
                        input_ids=semi_batch['input_ids'], 
                        attention_mask=semi_batch['attention_mask'], 
                        labels=semi_batch['labels'], 
                        mode='mlm'
                    )                    

                    loss = ce_loss + mlm_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                   
                    tr_ce_loss += ce_loss.item()
                    tr_mlm_loss += mlm_loss.item()
                    tr_loss += loss.item()

                    nb_tr_examples += batch['input_ids'].size(0)
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            ce_loss = tr_ce_loss / nb_tr_steps
            mlm_loss = tr_mlm_loss / nb_tr_steps

            eval_y_true, eval_y_pred = self.eval(args, get_feats=False)
            eval_score = round(accuracy_score(eval_y_true, eval_y_pred) * 100, 2)

            eval_results = {
                'Train_loss': round(loss, 6),
                'Train_ce_loss': round(ce_loss, 6),
                'Train_mlm_loss': round(mlm_loss, 6),
                'Eval_score': eval_score,
                'Best_score':best_eval_score,
                'Wait_epoch': wait
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch))
            for key in eval_results.keys():
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            # self.test(args)  # 作者还用测试集测试了

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
        save_path = os.path.join(args.output_dir, "best_pretrain_model.pt")
        torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Best pre-trained model saved to {save_path}")


if __name__ == "__main__":

    args = init_model()

    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.output_dir):
        raise RuntimeError(f"Failed to create output directory: {args.output_dir}")
    
    log_path = os.path.join(args.output_dir, "pretrain.log")


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
    manager = PretrainManager(args, data_processor, logger_name='Discovery')

    manager.train(args)