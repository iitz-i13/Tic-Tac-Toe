import numpy as np

class Node:
    '''ある1状態の探索結果を保存するクラス'''
    def __init__(self, p, v):
        self.p, self.v = p, v
        self.n, self.q_sum = np.zeros_like(p), np.zeros_like(p)
        self.n_all, self.q_sum_all = 1, v / 2 # 事前分布(見解が分かれる点)

    def update(self, action, q_new):
        # 行動のスタッツを更新
        self.n[action] += 1
        self.q_sum[action] += q_new

        # ノード全体のスタッツも更新
        self.n_all += 1
        self.q_sum_all += q_new