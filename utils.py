import random, os, copy
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# 匈牙利对齐算法
def hungray_alignment(y_true, y_pred):

    # 确定方阵维度
    D = max(y_pred.max(), y_true.max()) + 1
    # 构造混淆矩阵，w[i, j]表示预测标签为i、真实标签为j的样本出现次数
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # 代价矩阵cost = w.max() - w
    # 匈牙利算法linear_sum_assignment返回 行索引row_ind 和 列索引col_ind
    # ind: [min(D_pred, D_true), 2]
    # w: [num_pred_labels, num_true_labels] -> [D, D]
    ind = np.transpose(np.asarray(linear_sum_assignment(w.max()- w)))
    return ind, w

# 聚类评价指标    
def clustering_score(y_true, y_pred):
    """
    Return:
        dict: {'ACC', 'ARI', 'NMI'}
    """

    # clustering accuracy score
    ind, w = hungray_alignment(y_true=y_true, y_pred=y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    return {
        'ACC': round(acc*100, 2), 
        'ARI': round(adjusted_rand_score(y_true, y_pred)*100, 2), 
        'NMI': round(normalized_mutual_info_score(y_true, y_pred)*100, 2)
    }


def save_model(args, model, epoch, mode=None):
    model_path = os.path.join(args.output_dir, '{}_models_{}'.format('pretrain' if mode == 'pretrain' else 'cl', args.known_cls_ratio))
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # eg: results/cl_models_0.25/epoch_10.pt
    model_file = os.path.join(model_path, f'epoch_{epoch}.pt')
    model_dict = {
        'epoch': epoch, 
        'model_state_dict': model.state_dict(), 
    }

    torch.save(model_dict, model_file)


def save_results(args, test_results):

    pred_laobel_pth = os.path.join(args.output_dir, 'y_pred.npy')
    np.save(pred_laobel_pth, test_results['y_pred'])
    true_label_pth = os.path.join(args.output_dir, 'y_true.npy')
    np.save(true_label_pth, test_results['y_true'])

    del test_results['y_true']
    del test_results['y_pred']

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    var = [args.dataset, args.known_cls_ratio, args.num_labels, args.seed]
    names = ['dataset', 'KCR', 'num_labels', 'seed']
    vars_dict = {k: v for k, v in zip(names, var)}
    results = dict(test_results, **vars_dict)
    keys = list(results.keys())
    values = list(results.values())
    results_path = os.path.join(args.outputs_dir, "train_results")

    if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:
        ori = []
        ori.append(values)
        df1 = pd.DataFrame(ori, columns=keys)
        df1.to_csv(results, index=False)
    else:
        df1 = pd.read_csv(results_path)
        new = pd.DataFrame(results, index=[1])
        df1 = pd.concat([df1, new], ignore_index=True)
        df1.to_csv(results_path, index=False)
    
    data_diagram = pd.read_csv(results_path)

    print("test_results", data_diagram)

def mask_tokens(inputs, tokenizer, special_tokens_mask=None, mlm_probability=0.15):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    #https://github.com/huggingface/transformers/blob/master/src/transformers/data/data_collator.py#L70
    """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    probability_matrix[torch.where(inputs==0)] = 0.0
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

class view_generator:
    def __init__(self, tokenizer, rtr_prob, seed):
        set_seed(seed)
        self.tokenizer = tokenizer
        self.rtr_prob = rtr_prob
    
    def random_token_replace(self, ids):
        mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        # ids, _ = mask_tokens(ids, self.tokenizer, mlm_probability=0.25)
        ids, _ = mask_tokens(ids, self.tokenizer, mlm_probability=self.rtr_prob)
        random_words = torch.randint(len(self.tokenizer), ids.shape, dtype=torch.long)
        indices_replaced = torch.where(ids == mask_id)
        ids[indices_replaced] = random_words[indices_replaced]
        return ids

    def shuffle_tokens(self, ids):
        view_pos = []
        for inp in torch.unbind(ids):
            new_ids = copy.deepcopy(inp)
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(inp, already_has_special_tokens=True)
            sent_tokens_inds = np.where(np.array(special_tokens_mask) == 0)[0]
            inds = np.arange(len(sent_tokens_inds))
            np.random.shuffle(inds)
            shuffled_inds = sent_tokens_inds[inds]
            inp[sent_tokens_inds] = new_ids[shuffled_inds]
            view_pos.append(new_ids)
        view_pos = torch.stack(view_pos, dim=0)
        return view_pos