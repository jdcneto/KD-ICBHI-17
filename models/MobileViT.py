import torch
import torch.nn as nn
from utils import logMel
import torchaudio.transforms as T
from transformers import MobileViTForImageClassification


class MBViT(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-x-small")
        model.classifier = nn.Sequential(model.classifier, nn.Linear(1000, num_classes))

        self.model = model
        self.mel = logMel()
        self.spec_trf = nn.Sequential(T.FrequencyMasking(freq_mask_param=16),
                                      T.TimeMasking(time_mask_param=92))

    def forward(self, x):
        x = self.mel(x)
        if self.training:
            x = x.transpose(2,3)
            x = self.spec_trf(x)
            x = x.transpose(2,3)
        x = torch.cat((x,x,x), dim=1)

        x = self.model(x)
        x = x.logits    
        return x