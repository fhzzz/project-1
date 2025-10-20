from transformers import AutoModelForMaskedLM, AutoConfig
import torch.nn as nn
import torch.nn.functional as F

class PretrainBert(nn.Module):

    def __init__(self, model, num_labels=2):
        super(PretrainBert, self).__init__()

        self.backbone = AutoModelForMaskedLM.from_pretrained(model)
        self.config = AutoConfig.from_pretrained(model)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)


    def forward(self, input_ids=None, attention_mask=None, labels=None, mode=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels if mode == 'mlm' else None,
            output_hidden_states=True          
        )

        if mode == "mlm":
            return outputs.loss

        cls_emb = outputs.hidden_states[-1][:, 0]
        cls_emb = self.dropout(cls_emb)
        logits = self.classifier(cls_emb)

        if mode == "cls":
            return logits
        else:                       
            return cls_emb, logits
        

class Bert(nn.Module):

    def __init__(self, model, head_feat_dim):
        super(Bert, self).__init__()

        self.backbone = AutoModelForMaskedLM.from_pretrained(model)
        self.config = AutoConfig.from_pretrained(model)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(inplace=True), 
            nn.Dropout(p=0.1), 
            nn.Linear(self.config.hidden_size, head_feat_dim)
        )
        
    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                mode='feature_ext'):

        outputs = self.backbone(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                                output_hidden_states=True)

        # 统一先取最后一层 hidden
        last_hidden = outputs.hidden_states[-1]          # [B, L, H]
        mask = attention_mask.unsqueeze(-1).float()      # [B, L, 1]
        sent = (last_hidden * mask).sum(1) / (mask.sum(1) + 1e-8)  # mean-pooling

        if mode == 'feature_ext':
            return F.normalize(sent, p=2, dim=1)         

        elif mode == 'simple_forward':
            sent = self.dropout(sent)
            return F.normalize(self.head(sent), dim=1)   