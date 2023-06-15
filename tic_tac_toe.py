import numpy as np

class TicTacToe(object):
    '''○×ゲームの盤面実装'''
    X, Y = 'ABC',  '123'
    C = {0: '_', BLACK: 'O', WHITE: 'X'}

    def __init__(self):
        self.board = np.zeros((3, 3)) # (x, y)
        self.color = 1
        self.win_color = 0
        self.record = []

    def action2str(self, a):
        return self.X[a // 3] + self.Y[a % 3]

    def str2action(self, s):
        return self.X.find(s[0]) * 3 + self.Y.find(s[1])

    def record_string(self):
        return ' '.join([self.action2str(a) for a in self.record])

    def __str__(self):
        # 表示
        s = '   ' + ' '.join(self.Y) + '\n'
        for i in range(3):
            s += self.X[i] + ' ' + ' '.join([self.C[self.board[i, j]] for j in range(3)]) + '\n'
        s += 'record = ' + self.record_string()
        return s

    def play(self, action):
        # 行動で状態を進める関数
        # action は board 上の位置 (0 ~ 8) または行動系列の文字列
        if isinstance(action, str):
            for astr in action.split():
                self.play(self.str2action(astr))
            return self

        x, y = action // 3, action % 3
        self.board[x, y] = self.color

        # 3つ揃ったか調べる
        if self.board[x, :].sum() == 3 * self.color \
          or self.board[:, y].sum() == 3 * self.color \
          or (x == y and np.diag(self.board, k=0).sum() == 3 * self.color) \
          or (x == 2 - y and np.diag(self.board[::-1,:], k=0).sum() == 3 * self.color):
            self.win_color = self.color

        self.color = -self.color
        self.record.append(action)
        return self

    def terminal(self):
        # 終端状態かどうか返す
        return self.win_color != 0 or len(self.record) == 3 * 3

    def terminal_reward(self):
        # 終端状態での勝敗による報酬を返す
        return self.win_color if self.color == BLACK else -self.win_color

    def legal_actions(self):
        # 可能な行動リストを返す
        return [a for a in range(3 * 3) if self.board[a // 3, a % 3] == 0]

    def feature(self):
        # ニューラルネットに入力する状態表現を返す
        return np.stack([self.board == self.color, self.board == -self.color]).astype(np.float32)