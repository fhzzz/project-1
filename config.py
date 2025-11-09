from argparse import ArgumentParser

def init_model():

    parser = ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="Random seed")

    # 文件相关
    parser.add_argument("--data_dir", default="data", type=str, help="")
    parser.add_argument("--dataset", default="clinc", type=str, help="")
    parser.add_argument("--output_dir", default='results', type=str, help="")

    parser.add_argument("--known_cls_ratio", default=0.25, type=float, help="")
    parser.add_argument("--num_labels", default=150, type=int, help="")
    parser.add_argument("--feat_dim", default=128, type=int, help="")
    # tokenizer
    parser.add_argument("--max_length", default=None, type=int, help="")
    

    # 模型相关
    parser.add_argument("--model", default='./bert_model', help="")
    

    # 训练超参数相关
    parser.add_argument("--num_pretrain_epochs", default=100, type=int, help="")
    parser.add_argument("--num_train_epochs", default=50, type=int, help="")

    parser.add_argument("--pretrain_batch_size", default=16, type=int, help="")
    parser.add_argument("--train_batch_size", default=32, type=int, help="")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="")
    parser.add_argument("--test_batch_size", default=64, type=int, help="")
    
    parser.add_argument("--lr_pre", default=5.0e-5, type=float, help="")
    parser.add_argument("--lr", default=1.0e-5, type=float, help="")
    parser.add_argument("--wait_patient", default=10, type=int, help="Early stop patience")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="")


    # # UncertaintySector
    # parser.add_argument("--rho", default=1.0, type=float, help="")
    # parser.add_argument("--K", default=3, type=int, help="")
    # parser.add_argument("--alpha", default=1.0, type=float, help="")
    # parser.add_argument("--temperature", default=0.07, type=float, help="")

    # 对比损失
    parser.add_argument("--k_neg", type=int, default=2, help="Number of negative samples per positive pair.")
    
    # 调用LLM
    parser.add_argument("--temperature", default=0.07, type=float, help="")
    parser.add_argument("--model_name", default="chatgpt-3.5-turbo", type=str, help="")
    parser.add_argument("--update_per_epochs", default=3, type=int, help="")
    parser.add_argument("--max_retry", default=5, type=int, help="")

    # 其他
    parser.add_argument("--max_seq_length", type=int, help="")
    parser.add_argument("--save_results", action="store_true", help="")
    args = parser.parse_args()   # 关键：一定要 .parse_args()
    return args                  # 返回的是 Namespace，不是 parser