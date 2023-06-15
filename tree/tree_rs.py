import numpy as np
import time, copy
import torch
import torch.nn as nn
from node import Node
from net import Net

class Tree_RS:
    '''探索木を保持してモンテカルロ木探索を行うクラス'''
    def __init__(self, net):
        self.net = net
        self.nodes = {}
        self.count = 0
    
    def search(self, state, depth):
        # 終端状態の場合は末端報酬を返す
        if state.terminal():
            return state.terminal_reward()

        # まだ未到達の状態はニューラルネットを計算して推定価値を返す
        key = state.record_string()
        if key not in self.nodes:
            p, v = self.net.predict(state)
            self.nodes[key] = Node(p, v)
            return v

        # 到達済みの状態はバンディットで行動を選んで状態を進める
        node = self.nodes[key]
        p = node.p
        if depth == 0:
            # ルートノード(現局面)では方策にノイズを加える
            p = 0.75 * p + 0.25 * np.random.dirichlet([0.15] * len(p))

        best_action, best_RS = None, -float('inf')
        R = 0.6
        c=0
        legal_actions =state.legal_actions()
        for action in legal_actions:
            n, q_sum = 1 + node.n[action], node.q_sum_all / node.n_all + node.q_sum[action]
            #Q_sum_それまでのノードの報酬/ノードの数＋この腕の報酬(勝率)
            #ここまでの腕単体だけを見たときの価値?node.q_sum[action]
            # ucb = q_sum / n + 2.0 * np.sqrt(node.n_all) * p[action] / n # PUCTの式
            RS = n*(q_sum/n - R)/node.n_all # RSの式
            if RS > best_RS:
              best_action, best_RS = action, RS
            if RS > 0:
              best_action, best_RS = action, RS
              break
            
        if not best_action == None:
            state.play(best_action)
        else:
            self.count = self.count +1
        if self.count  > 2:
            state.win_color = state.pass_drop()
            self.count  = 2
        q_new = -self.search(state, depth + 1) # 1手ごとの手番交代を想定
        node.update(best_action, q_new)

        return q_new

    def think(self, state, num_simulations, temperature = 0, show=False):
        # 探索のエンドポイント
        if show:
            print(state)
        start, prev_time = time.time(), 0
        for _ in range(num_simulations):
            self.search(copy.deepcopy(state), depth=0)

            # 1秒ごとに探索結果を表示
            if show:
                tmp_time = time.time() - start
                if int(tmp_time) > int(prev_time):
                    prev_time = tmp_time
                    root, pv = self.nodes[state.record_string()], self.pv(state)
                    print('%.2f sec. best %s. q = %.4f. n = %d / %d. pv = %s'
                          % (tmp_time, state.action2str(pv[0]), root.q_sum[pv[0]] / root.n[pv[0]],
                             root.n[pv[0]], root.n_all, ' '.join([state.action2str(a) for a in pv])))

        #  訪問回数で重みつけた確率分布を返す
        n = root = self.nodes[state.record_string()].n + 1
        n = (n / np.max(n)) ** (1 / (temperature + 1e-8))
        return n / n.sum()

    def pv(self, state):
        # 最善応手列（読み筋）を返す
        s, pv_seq = copy.deepcopy(state), []
        while True:
            key = s.record_string()
            if key not in self.nodes or self.nodes[key].n.sum() == 0:
                break
            best_action = sorted([(a, self.nodes[key].n[a]) for a in s.legal_actions()], key=lambda x: -x[1])[0][0]
            pv_seq.append(best_action)
            s.play(best_action)
        return pv_seq