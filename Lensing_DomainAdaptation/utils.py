import torch
import random
import numpy as np
import os

def set_seed(seed):
    #Sets the seed for Reproducibility
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class HPAMS:
    def __init__(self):
        #Pretraining_model_hyperparameters
        self.pretraining_epochs = 7
        self.pretraining_warmup_epochs = 3
        self.pretraining_learning_rate = 1e-4
        self.pretraining_weight_decay = 1e-5

        # ADDA_MODEL_HYperparameters
        self.adversarial_epochs = 5
        self.adversarial_warmup_epochs = 2
        self.discriminator_learning_rate = 1e-4
        self.discriminator_weight_decay = 1e-5
        self.target_learning_rate = 1e-6
        self.targetweight_decay = 1e-5
