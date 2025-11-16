from transformers import AutoModelForMaskedLM, AutoConfig
import torch
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

    def __init__(self, model, feat_dim=128):
        super(Bert, self).__init__()

        self.backbone = AutoModelForMaskedLM.from_pretrained(model)
        self.config = AutoConfig.from_pretrained(model)

        # 评估分类准确率
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size) # Pooling-mean
        # self.activation = nn.Tanh()
        # self.dropout = nn.Dropout(p=0.1)
        # self.classifier = nn.Linear(config.hidden_size, num_labels)

        # 对比损失投影头
        self.head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(inplace=True), 
            nn.Dropout(p=0.1), 
            nn.Linear(self.config.hidden_size, feat_dim)
        )
        
    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, 
                labels=None, label=None, u_threshold=None, l_threshold=None,  
                mode=None, semi=False):        

        outputs = self.backbone(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids, 
                                labels=None, 
                                output_hidden_states=True)

        # # 统一先取最后一层 hidden, mean pooling
        # last_hidden = outputs.hidden_states[-1]          # [B, L, H]
        # mask = attention_mask.unsqueeze(-1).float()      # [B, L, 1]
        # sent_embed = (last_hidden * mask).sum(1) / (mask.sum(1) + 1e-8)  # mean-pooling

        # [CLS]token
        sent_embed = outputs.hidden_states[-1][:, 0]
        feats = F.normalize(self.head(sent_embed), dim=1)
        if mode == "train":
            eps = 1e-10
            logits_norm = F.normalize(feats, p=2, dim=1)
            sim_mat = torch.matmul(logits_norm, logits_norm.t())  # [B, B]

            label_mat = label.view(-1, 1) - label.view(1, -1)  # [B, B]
            label_mat[label_mat != 0] = -1  # dis-pair
            label_mat[label_mat == 0] = 1  # sim-pair
            label_mat[label_mat == -1] = 0  # dis-pair -> 0

            if not semi:
                pos_mask = (label_mat > u_threshold).float()
                neg_mask = (label_mat < l_threshold).float()
                pos_loss = -torch.log(torch.clamp(sim_mat, eps, 1.0)) * pos_mask
                neg_loss = -torch.log(torch.clamp(1 - sim_mat, eps, 1.0)) * neg_mask
                loss = (pos_loss.mean() + neg_loss.mean()) * 5
                return loss
            else:
                label_mat[label == -1, :] = -1
                label_mat[:, label == -1] = -1
                label_mat[label_mat == 0] = 0
                label_mat[label_mat == 1] = 1

                pos_mask = (sim_mat > u_threshold).float()
                neg_mask = (sim_mat < l_threshold).float()
                pos_mask[label_mat == 1] = 1
                neg_mask[label_mat == 0] = 1

                pos_loss = -torch.log(torch.clamp(sim_mat, eps, 1.0)) * pos_mask
                neg_loss = -torch.log(torch.clamp(1 - sim_mat, eps, 1.0)) * neg_mask
                loss = pos_loss.mean() + neg_loss.mean() + u_threshold - l_threshold
                return loss
        else:        
            return feats
    

class BertForConstrainClustering(nn.Module):

    def __init__(self, model, num_labels):
        super(BertForConstrainClustering, self).__init__()

        self.backbone = AutoModelForMaskedLM.from_pretrained(model)
        self.config = AutoConfig.from_pretrained(model)
        
        # Train components
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)  # Pooling-mean
        self.cls_layer = nn.Sequential(
        nn.Tanh(), 
        nn.Dropout(self.config.hidden_dropout_prob), 
        nn.Linear(self.config.hidden_size, num_labels))        

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, 
                u_threshold=None, l_threshold=None, labels=None, 
                mode=None, semi=False):

        eps = 1e-10
        # Extract features from the backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, 
                                token_type_ids=token_type_ids, output_hidden_states=True)
        encoded_layer_12 = outputs.hidden_states[-1]  # Get the last hidden layer
        pooled_output = self.dense(encoded_layer_12.mean(dim=1))  # Mean pooling
        logits = self.cls_layer(pooled_output)

        if mode == "train":
            logits_norm = F.normalize(logits, p=2, dim=1)
            sim_mat = torch.matmul(logits_norm, logits_norm.t())  # [B, B]

            label_mat = labels.view(-1, 1) - labels.view(1, -1)  # [B, B]
            label_mat[label_mat != 0] = -1  # dis-pair
            label_mat[label_mat == 0] = 1  # sim-pair
            label_mat[label_mat == -1] = 0  # dis-pair -> 0

            if not semi:
                pos_mask = (label_mat > u_threshold).float()
                neg_mask = (label_mat < l_threshold).float()
                pos_loss = -torch.log(torch.clamp(sim_mat, eps, 1.0)) * pos_mask
                neg_loss = -torch.log(torch.clamp(1 - sim_mat, eps, 1.0)) * neg_mask
                loss = (pos_loss.mean() + neg_loss.mean()) * 5
                return loss
            else:
                label_mat[labels == -1, :] = -1
                label_mat[:, labels == -1] = -1
                label_mat[label_mat == 0] = 0
                label_mat[label_mat == 1] = 1

                pos_mask = (sim_mat > u_threshold).float()
                neg_mask = (sim_mat < l_threshold).float()
                pos_mask[label_mat == 1] = 1
                neg_mask[label_mat == 0] = 1

                pos_loss = -torch.log(torch.clamp(sim_mat, eps, 1.0)) * pos_mask
                neg_loss = -torch.log(torch.clamp(1 - sim_mat, eps, 1.0)) * neg_mask
                loss = pos_loss.mean() + neg_loss.mean() + u_threshold - l_threshold
                return loss

        # elif mode == "finetune":
        #     # Student-t 核计算软分配 q
        #     dist = torch.sum((logits.unsqueeze(1) - self.cluster_layer) ** 2, dim=2)  # [B, K]
        #     q = 1.0 / (1.0 + dist / self.alpha)
        #     q = q.pow((self.alpha + 1.0) / 2.0)
        #     q = (q.t() / torch.sum(q, dim=1)).t()  # 行归一化
        #     return logits, q

        else:  # inference
            return logits
        

