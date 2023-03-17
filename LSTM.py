import torch
import torch.nn as nn

class LSTM_model(nn.Module):
    def __init__(self, actions):
        super(LSTM_model, self).__init__()

        self.actions = actions

        self.lstm1 = nn.LSTM(132, 128, batch_first=True, bidirectional=False)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(128, 256, batch_first=True, bidirectional=False)
        self.dropout2 = nn.Dropout(0.2)
        self.lstm3 = nn.LSTM(256, 256, batch_first=True, bidirectional=False)
        self.bn = nn.BatchNorm1d(256)
        self.dense1 = nn.Linear(256, 256)
        self.dense2 = nn.Linear(256, 128)
        self.dense3 = nn.Linear(128, 64)
        self.dense4 = nn.Linear(64, actions.shape[0])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x, _ = self.lstm3(x[:, -1, :].unsqueeze(1))
        x = self.bn(x.squeeze())
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.softmax(x)

        return x
