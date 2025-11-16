from data_process import *
from config import *
from model import *
from losses import *
from utils import *
from cdac import *


args = init_model()
data_processor = PrepareData(args)
manager = CdacManager(args, data_processor)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clustering_score, _ = manager.evaluation(args)
print(clustering_score)