

class MemoryBank:
    
    def __init__(self, capacity, feature_dim, num_classes, relation_dim=64):

        self.capacity = capacity
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # 基础存储
        self.features = torch.zeros(capacity, feature_dim)
        