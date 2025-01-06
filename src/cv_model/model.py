from torch import nn
from torchvision import models


def get_model() -> nn.Module:

    fe = models.mobilenet_v3_large(pretrained=True)
    fe = nn.Sequential(*list(fe.children())[0][:])

    return fe


class Model(nn.Module):
    def __init__(self,
                 num_cats: int,
                 num_names: int):
        super().__init__()

        self.fe = get_model()

        self.aap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.cats = nn.Sequential(
            nn.Linear(960, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),

            nn.Linear(256, num_cats)
        )

        self.names = nn.Sequential(
            nn.Linear(960, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),

            nn.Linear(256, num_names)
        )
        
    def forward(self, x):

        features = self.flatten(self.aap(self.fe(x)))
        out_cats = self.cats(features)
        out_names = self.names(features)

        return out_cats, out_names