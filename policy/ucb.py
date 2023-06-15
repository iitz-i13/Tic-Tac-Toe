import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from net import Net
from node import Node
from tree.tree_ucb import Tree_UCB
from common import show_net
from common import train
from common import vs_random
from common import vs_player
from common import vs_random_result
from common import vs_ucb_result

from tic_tac_toe import TicTacToe as State

def train_AlphaZero():
    # 学習前なのでランダムな出力を得る
    show_net(Net(), State())#一戦行う

    # 初期ネットワークで探索を行う
    tree = Tree(Net())
    tree.think(State(), 1000, show=True)

    """# Main 文
    実際に学習と対戦の実行を行うところ
    """
    # ucbの学習
    sims = 1
    # 学習のための Self-play の試合数
    num_games = 100
    # 何回の試合ごとに学習するか
    num_train_steps = 1
    # 何回先読みするか
    num_simulations = 20

    net_ucb = Net()
    episodes = []
    result_distribution = {1:0, 0:0, -1:0}
    print('vs_random = ', sorted(vs_random(net_ucb).items()))
    for i in range(sims):
        print('sim=',i)
        # net3 == net_init #初期化 # Pythonのイコールはポインタ参照だったはずなのであやしい
        net3 = Net() #これでいいのでは？
        episodes = []
        result_distribution = {1:0, 0:0, -1:0}
        for g in range(num_games):
            # 1対戦のエピソード生成
            record, p_targets = [], []
            state = State()
            tree = Tree_UCB(net_ucb)
            temperature = 0.7 # 探索結果から方策のtargetを作るときの温度 (softmax 関数の温度パラメータ)
            while not state.terminal():
                p_target = tree.think(state, num_simulations, temperature)
                # 行動を推論した確立分布 (p_target) に従ってランダムに選んで進める
                action = np.random.choice(np.arange(len(p_target)), p=p_target)
                state.play(action)
                record.append(action)
                p_targets.append(p_target)
                temperature *= 0.8
            # 先手視点の報酬
            reward = state.terminal_reward() * (1 if len(record) % 2 == 0 else -1)
            result_distribution[reward] += 1
            episodes.append((record, reward, p_targets))
            if g % num_train_steps == 0:
                print('game ', end='')
            print(g, ' ', end='')    
            # ニューラルネットの学習
            if (g + 1) % num_train_steps == 0:
                # 先手勝、引き分け、後手勝ちの割合を表示
                print('generated = ', sorted(result_distribution.items()))
                net_ucb = train(episodes)
                print('vs_random = ', sorted(vs_random(net_ucb).items()))
            #model_save_name = 'FileName.pth'
            #path = F"./128/{model_save_name}"
            if g % 100 == 0:
                model_save_name = f'UCB.pth'
                torch.save(net_ucb.state_dict(), f"model/{model_save_name}")

    print('finished')

if __name__ == "__main__":
    train_AlphaZero()