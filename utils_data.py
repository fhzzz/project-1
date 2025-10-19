

class MemoryBank:
    
    def __init__(self, capacity, feature_dim, num_classes, relation_dim=64):

        self.capacity = capacity
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # 基础存储
        self.features = torch.zeros(capacity, feature_dim)
        self.targets = torch.zeros(capacity, dtype=torch.long)
        self.uncertainties = torch.zeros(capacity)

        # 关系矩阵
        self.relation_matrix = torch.zeros(capacity, capacity)
        self.targets = torch.zeros(capacity, dtype=torch.long)
        
        # 样本对管理
        self.uncertain_pairs = []
        self.confident_pairs = []
        self.pair_max_age = 1000

        # 索引管理

        # 配置参数
        self.uncert_threshold = uncert_thershold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._to_device(self.device)

    def _to_device(self, device):
        """将数据移动到指定设备"""
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.uncertainties = self.uncertainties.to(device)
        self.relation_matrix = self.relation_matrix.to(device)


class NeighborDataset:

    def __init__(self, dataset, indices, num_neighbors=None):
        super(NeighborDataset, self).__init__()

        self.dataset = dataset
        self.indices = indices  # Nearest neighbor indices (np.array, [len(dataset), 1])
        # 这里如果要根据LLM标注结果确定正类样本的话，
    



        