import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tic_tac_toe import TicTacToe as State

# ニューラルネットのパラメータ
num_filters = 16
num_blocks = 6

class Conv(nn.Module):
    def __init__(self, filters0, filters1, kernel_size, bn=False):
        super().__init__()
        self.conv = nn.Conv2d(filters0, filters1, kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(filters1)

    def forward(self, x):
        h = self.conv(x)
        if self.bn is not None:
            h = self.bn(h)
        return h


class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv = Conv(filters, filters, 3, True)

    def forward(self, x):
        return F.relu(x + (self.conv(x)))
        

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        state = State()
        self.input_shape = state.feature().shape
        self.board_size = self.input_shape[1] * self.input_shape[2]

        # filter 方向に拡張するネットワーク
        self.layer0 = Conv(self.input_shape[0], num_filters, 3, bn=True)
        # Res block を積み重ねている
        self.blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_blocks)])

        # 方策分布のネットワーク
        self.conv_p1 = Conv(num_filters, 4, 1, bn=True)
        self.conv_p2 = Conv(4, 1, 1)
        # 価値のネットワーク
        self.conv_v = Conv(num_filters, 4, 1, bn=True)
        self.fc_v = nn.Linear(self.board_size * 4, 1, bias=False)

    def forward(self, x):
        h = F.relu(self.layer0(x))
        for block in self.blocks:
            h = block(h)
        # 方策分布のネットワーク
        h_p = F.relu(self.conv_p1(h))
        h_p = self.conv_p2(h_p).view(-1, self.board_size)
        # 価値のネットワーク
        h_v = F.relu(self.conv_v(h))
        h_v = self.fc_v(h_v.view(-1, self.board_size * 4))
        # 方策と価値の二股になっている

        # value(状態価値)にtanhを適用するので負け -1 ~ 勝ち 1
        return F.softmax(h_p, dim=-1), torch.tanh(h_v)

    def predict(self, state):
        # 探索中に呼ばれる推論関数
        self.eval()
        x = torch.from_numpy(state.feature()).unsqueeze(0)
        with torch.no_grad():
            p, v = self.forward(x)
        return p.cpu().numpy()[0], v.cpu().numpy()[0][0]