import torch
import torch.nn as nn
import torch.nn.functional as F


class xvecTDNN(nn.Module):

    def __init__(self,feature_dim, num_lang, p_dropout):
        super(xvecTDNN, self).__init__()

        self.dropout = nn.Dropout(p=p_dropout)

        self.tdnn1 = nn.Conv1d(in_channels=feature_dim, out_channels=512, kernel_size=5, dilation=1)
        self.bn1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)

        self.tdnn2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, dilation=2)
        self.bn2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)

        self.tdnn3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7, dilation=3)
        self.bn3 = nn.BatchNorm1d(512, momentum=0.1, affine=False)

        self.tdnn4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        self.bn4 = nn.BatchNorm1d(512, momentum=0.1, affine=False)

        self.tdnn5 = nn.Conv1d(in_channels=512, out_channels=1500, kernel_size=1, dilation=1)
        self.bn5 = nn.BatchNorm1d(1500, momentum=0.1, affine=False)

        self.fc6 = nn.Linear(3000,512)
        self.bn6 = nn.BatchNorm1d(512, momentum=0.1, affine=False)

        self.fc7 = nn.Linear(512,512)
        self.bn7 = nn.BatchNorm1d(512, momentum=0.1, affine=False) #momentum=0.5 in asv-subtools

        self.fc8 = nn.Linear(512,num_lang)

    def forward(self, x, eps=1e-5):
        # Note: x must be (batch_size, feat_dim, chunk_len)
        # print("\tIn Model: input size", x.size())
        # x = self.dropout(x)
        x = self.bn1(F.relu(self.tdnn1(x)))
        x = self.dropout(x)
        x = self.bn2(F.relu(self.tdnn2(x)))
        x = self.dropout(x)
        x = self.bn3(F.relu(self.tdnn3(x)))
        x = self.dropout(x)
        x = self.bn4(F.relu(self.tdnn4(x)))
        x = self.dropout(x)
        x = self.bn5(F.relu(self.tdnn5(x)))

        if self.training:
            shape = x.size()
            noise = torch.cuda.FloatTensor(shape)
            torch.randn(shape, out=noise)
            x += noise*eps

        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        # print("pooling", stats.size())
        x = self.bn6(F.relu(self.fc6(stats)))
        x = self.dropout(x)
        x = self.bn7(F.relu(self.fc7(x)))
        x = self.dropout(x)
        output = self.fc8(x)
        # print("\toutput size", output.size())
        return output

class xvec_extractor(nn.Module):
    def __init__(self,feature_dim):
        super(xvec_extractor, self).__init__()
        self.tdnn1 = nn.Conv1d(in_channels=feature_dim, out_channels=512, kernel_size=5, dilation=1)
        self.bn1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.tdnn2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, dilation=2)
        self.bn2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.tdnn3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7, dilation=3)
        self.bn3 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.tdnn4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        self.bn4 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.tdnn5 = nn.Conv1d(in_channels=512, out_channels=1500, kernel_size=1, dilation=1)
        self.bn5 = nn.BatchNorm1d(1500, momentum=0.1, affine=False)
        self.fc6 = nn.Linear(3000,512)
        self.bn6 = nn.BatchNorm1d(512, momentum=0.1, affine=False)

    def forward(self, x):
        # Note: x must be (batch_size, feat_dim, chunk_len)
        x = self.bn1(F.relu(self.tdnn1(x)))
        x = self.bn2(F.relu(self.tdnn2(x)))
        x = self.bn3(F.relu(self.tdnn3(x)))
        x = self.bn4(F.relu(self.tdnn4(x)))
        x = self.bn5(F.relu(self.tdnn5(x)))
        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        output = self.fc6(stats)
        return output
