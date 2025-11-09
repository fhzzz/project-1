from utils import *
from config import *

from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset, ClassLabel, concatenate_datasets
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, Dataset
import os, logging, copy

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import euclidean_distances, pairwise_distances_argmin_min

class PrepareData:

    def __init__(self, args, logger_name="Data Processing"):

        set_seed(args.seed)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:   # 避免重复
            self.logger.addHandler(logging.StreamHandler())        

        self.data_dir = os.path.join(args.data_dir, args.dataset)
        self.logger.info(f"data_dir:{self.data_dir}")

        max_seq_lengths = {'clinc':30, 'stackoverflow':45,'banking':55}
        args.max_seq_length = max_seq_lengths[args.dataset]
        self.max_seq_length = args.max_seq_length
        self.logger.info(f"max_seq_length: {self.max_seq_length}")

        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.tokenized_datasets = self.get_tokenized(self.data_dir)        
        self.all_ordered, self.known_cls_list, self.unk_cls_list = self.get_labels(args, self.tokenized_datasets["train"])
        self.logger.info(f"all labels={len(self.all_ordered)}, known labels={len(self.known_cls_list)}")
        self.tokenized_datasets = self.label2id(self.tokenized_datasets, list=self.all_ordered)
        self.logger.info(f"处理完成后的tokenized_datasets: {self.tokenized_datasets}")

        self.train_labeled_samples, self.train_semi_samples, self.eval_known_samples = self.get_samples(self.tokenized_datasets, args)
        self.test_samples = self.tokenized_datasets["test"]

        self.logger.info(f"训练集有标签样本数量：{len(self.train_labeled_samples)}")
        self.logger.info(f"训练集半监督样本数量：{len(self.train_semi_samples)}")
        self.logger.info(f"测试集样本数量: {len(self.test_samples)}")

        # DataLoader: 预训练阶段
        self.train_labeled_dataloader = self.get_dataloader(self.train_labeled_samples, args, mode=None)  # shuffle=True
        self.train_mlm_dataloader = self.get_dataloader(self.train_semi_samples, args, mode='mlm')  # shuffle=True

        self.eval_known_dataloader = DataLoader(dataset=self.eval_known_samples, 
                                                batch_size=args.eval_batch_size, shuffle=False)
        
        # DataLoader: 对比学习阶段, 必须使用顺序采样器
        # train_semi_sampler = SequentialSampler(self.train_semi_samples)
        # self.train_semi_dataloader = DataLoader(dataset=self.train_semi_samples, batch_size=args.train_batch_size, 
        #                                         sampler=train_semi_sampler)

        # self.logger.info("cdac training, shuffle=True")
        # self.train_semi_dataloader = DataLoader(dataset=self.train_semi_samples, batch_size=args.pretrain_batch_size, 
        #                                         shuffle=True)
                                               
        # 测试集同样采样了顺序采样器
        test_sampler = SequentialSampler(self.test_samples)
        self.test_dataloader = DataLoader(dataset=self.test_samples, batch_size=args.test_batch_size, 
                                          sampler=test_sampler)
        
        # 训练集数据示例
        # batch = next(iter(self.train_labeled_dataloader))
        # first_sample = {}
        # for key, value in batch.items():
        #     first_sample[key] = value[0]  # 取每个张量的第一个元素

        # self.logger.info("第一个标签样本的完整数据:")
        # self.logger.info(first_sample)


    def get_tokenized(self, data_dir):

        data_files = {
            "train": os.path.join(data_dir, "train.tsv"), 
            "validation": os.path.join(data_dir, "validation.tsv"), 
            "test": os.path.join(data_dir, "test.tsv")
            }
        raw_datasets = load_dataset("csv", data_files=data_files)
        print(raw_datasets)
        print(raw_datasets["train"][0])

        self.index_to_text = {
            split: {i: example['text'] for i, example in enumerate(raw_datasets[split])} \
                for split in ["train", "validation", "test"]
        }
        
        def tokenized_function(example):
            return self.tokenizer(
                example["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_length
            )
        
        tokenized_datasets = raw_datasets.map(tokenized_function, batched=True).remove_columns("text")
        tokenized_datasets = tokenized_datasets.rename_column('Unnamed: 0', 'index')
        tokenized_datasets.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "index", "label"])

        return tokenized_datasets
    
    def get_labels(self, args, dataset):
        # 此时label是文本，未经过映射
        all_cls_list = dataset.unique("label")
        n_known_cls = round(len(all_cls_list) * args.known_cls_ratio)
        known_cls_list = np.random.choice(all_cls_list, n_known_cls, replace=False)
        unk_cls_list = [label for label in all_cls_list if label not in known_cls_list]
        all_ordered = list(known_cls_list) + list(unk_cls_list)

        return all_ordered, known_cls_list, unk_cls_list

    def label2id(self, datasetdict, list):
        full_label_feature = ClassLabel(names=list)
        datasetdict = datasetdict.cast_column(column='label', feature=full_label_feature)
        
        return datasetdict
    
    def get_samples(self, datasetdict, args):

        known_intent_datasets = datasetdict["train"].filter(
            lambda x: x['label'] < len(self.known_cls_list)
        )

        eval_known_samples = datasetdict["validation"].filter(
            lambda x: x["label"] < len(self.known_cls_list)
        )

        known_intent_datasets = known_intent_datasets.train_test_split(test_size=0.9, stratify_by_column="label", seed=args.seed)
        train_labeled_samples = known_intent_datasets["train"]
        # print(f"trian_labeled_samples:{train_labeled_samples[0]}")

        
        # 处理无标签数据: label设为-1 + 合并
        label_idx = [ex['index'] for ex in train_labeled_samples]
        # print(label_idx[:5])
        train_unlabel_samples = datasetdict["train"].filter(lambda x: x['index'] not in label_idx)
        # print(f"train_unlabel_samples:{train_unlabel_samples[0]}")

        # 注意，这里是需要赋值的
        train_unlabel_samples = train_unlabel_samples.map(lambda x: {"label": -1}, batched=False)
        # print(f"after map:{train_unlabel_samples[0]}")
        train_semi_samples = concatenate_datasets([train_labeled_samples, train_unlabel_samples])
        # print(f"semi:{train_semi_samples[:3]}")

        return train_labeled_samples, train_semi_samples, eval_known_samples
    
    def get_dataloader(self, dataset, args, mode=None):

        mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=True, 
            mlm_probability=0.15
        )
        return DataLoader(dataset, 
                          batch_size=args.pretrain_batch_size, 
                          shuffle=True, 
                          collate_fn=mlm_collator if mode == 'mlm' else None)


if __name__ == '__main__':

    args = init_model()

    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.output_dir):
        raise RuntimeError(f"Failed to create output directory: {args.output_dir}")
    
    log_path = os.path.join(args.output_dir, "data_process.log")

    logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()   # 控制台
    ]
    )
    data_processer = PrepareData(args=args, logger_name="Data Processing")