import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tic_tac_toe import TicTacToe as State
from net import Net
from tree.tree_rs import Tree_RS
from tree.tree_ucb import Tree_UCB

#ニューラルネットの学習に関するパラメータ
batch_size = 32
num_epochs = 30

def show_net(net, state):
    '''方策 (p) と 状態価値 (v) を表示'''
    print(state) # 環境を表示
    p, v = net.predict(state)
    print('p = ')#pを表示
    print((p *1000).astype(int).reshape((-1, *net.input_shape[1:3])))
    #方策を表示
    print('v = ', v)
    print()

def gen_target(ep):
    #ニューラルネットの学習用 input, targets を生成
    turn_idx = np.random.randint(len(ep[0]))
    state = State()#環境の呼び出し
    for a in ep[0][:turn_idx]:#試合が終わるまでループ
        state.play(a)#試合を行う
    v = ep[1]
    # ニューラルネットに入力する状態表現を返す
    return state.feature(), ep[2][turn_idx], [v if turn_idx % 2 == 0 else -v]

def train(episodes):#トレーニング用の関数
    net = Net()#ニューラルネットの呼び出し
    optimizer = optim.SGD(net.parameters(), lr=1e-3, weight_decay=1e-4, momentum=0.75)#SGDでパラメータの更新
    for epoch in range(num_epochs):#ループ
        p_loss_sum, v_loss_sum = 0, 0#pとvの合計
        net.train()# trainの呼び出し
        for i in range(0, len(episodes), batch_size):
            x, p_target, v_target = zip(*[gen_target(episodes[np.random.randint(len(episodes))]) for j in range(batch_size)])# pとvの基準？
            x = torch.FloatTensor(np.array(x))# xを多次元配列のデータ構造で設定
            p_target = torch.FloatTensor(np.array(p_target))# p_tergetを多次元配列のデータ構造で設定
            v_target = torch.FloatTensor(np.array(v_target))# v_tergetを多次元配列のデータ構造で設定
            
            p, v = net(x)# p,vをnetを用いて表す？
            p_loss = torch.sum(-p_target * torch.log(p))# p_lossの計算
            v_loss = torch.sum((v_target - v) ** 2)# vのlossの計算

            p_loss_sum += p_loss.item()# p_loss_sumの計算
            v_loss_sum += v_loss.item()# v_loss_sumの計算

            optimizer.zero_grad()# 勾配を初期化
            (p_loss + v_loss).backward()# 出力から微分
            optimizer.step()# 損失をモデルに反映される

        for param_group in optimizer.param_groups:# ループさせる
            param_group['lr'] *= 0.85# param_groupの更新
    # p,vのlossをprintさせる
    print('p_loss %f v_loss %f' % (p_loss_sum / len(episodes), v_loss_sum / len(episodes)))
    return net

def vs_random(net, n=100):
    results = {}
    legal_actions = []
    count = 0
    for i in range(n):
        first_turn = i % 2 == 0 # どちらのプレイヤーのターンか確定させる
        turn = first_turn # 確定
        state = State() # stateの呼び出し
        while not state.terminal():
            if turn:
                legal_actions = state.legal_actions()
                if legal_actions == -1 :
                    count = count +1 
                    #print(count) 
                    if count > 2:
                        state.win_color = state.pass_drop()
                        break 
                else:
                    if len(legal_actions) ==0 or legal_actions == None:
                        count = count +1 
                        state.color = - state.color
                        #print(count) 
                        if count > 2:
                            state.win_color = state.pass_drop()
                            break 
                    else:
                        if  len(legal_actions) == 1:
                            action = np.random.choice(legal_actions)
                            state.play(action)
                            count = 0
                        else:
                            p, _ = net.predict(state)#方策を呼び出す
                            #print(f"p: {p}")
                            action = sorted([(a, p[a]) for a in legal_actions], key=lambda x:-x[1])[0][0]#方策に従って手を選ぶ
                            state.play(action)#実際にアクションを入力して打つ
                            count = 0
            else:
                legal_actions = state.legal_actions()
                if len(legal_actions) == 0 or legal_actions == None :  
                    count = count +1  
                    #print(count) 
                    state.color = - state.color
                    if count > 2:
                     break 
                else:
                    action = np.random.choice(legal_actions)#ランダムで盤面から手を選ぶ
                    state.play(action)#実際にアクションを入力して打つ
            turn = not turn
        r = state.terminal_reward() if turn else -state.terminal_reward()#どちらが勝ったのか判定し報酬を渡す
        results[r] = results.get(r, 0) + 1#試合が終わった時ターンによって勝ち数を増やす
    return results

def vs_player(net, n=1):
    results = {}
    for i in range(n):
        first_turn = i % 2 == 0
        turn = first_turn
        state = State()
        while not state.terminal():
            if turn:
                p, _ = net.predict(state)
                action = sorted([(a, p[a]) for a in state.legal_actions()], key=lambda x:-x[1])[0][0]## vs_randomと同じ
            else:
                action = np.random.choice(state.legal_actions())
            state.play(action)
            turn = not turn
        r = state.terminal_reward() if turn else -state.terminal_reward()
    return results # vs_randomと同じ(コピーミス）

def vs_random_result(net, n=100):
    results = {}
    result1 = 0
    for i in range(n):
        first_turn = i % 2 == 0
        turn = first_turn
        state = State()
        while not state.terminal():
            if turn:
                p, _ = net.predict(state)
                action = sorted([(a, p[a]) for a in state.legal_actions()], key=lambda x:-x[1])[0][0]
                # action = tree.bm(state, 10000)
            else:
            #   p, _ = net.predict(state)
            #   action = sorted([(a, p[a]) for a in state.legal_actions()], key=lambda x:-x[1])[0][0]
            #   action = tree.bm(state, 10000)
              action = np.random.choice(state.legal_actions())
            state.play(action)
            turn = not turn
        r = state.terminal_reward() if turn else -state.terminal_reward()
        results[r] = results.get(r, 0) + 1
        if r == 1:
          result1=result1 + 1
    return result1

def vs_ucb_result(net1,net2,n=100):
    results = {}
    result1 = 0
    tree1 = Tree(net1)
    tree2 = Tree_UCB(net2)
    
    for i in range(n):
        first_turn = i % 2 == 0
        turn = first_turn
        state = State()
        random_turn = 0
        while not state.terminal():
            if random_turn == 0:
              action = np.random.choice(state.legal_actions())
              state.play(action)
              
              random_turn = 1
            if turn:
              p, _ = net1.predict(state)
              action = sorted([(a, p[a]) for a in state.legal_actions()], key=lambda x:-x[1])[0][0]
            #   action = tree1.bm(state, 1000)
              state.play(action)
            else:
              p1, _ = net2.predict(state)
              action = sorted([(a, p1[a]) for a in state.legal_actions()], key=lambda x:-x[1])[0][0]
            #   action = tree2.bm(state, 1000)
              
              state.play(action)
            #   action = np.random.choice(state.legal_actions())
            #   p1, _ = net2.predict(state)
            #   action = sorted([(a, p1[a]) for a in state.legal_actions()], key=lambda x:-x[1])[0][0]
            # state.play(action)
            turn = not turn
        r = state.terminal_reward() if turn else -state.terminal_reward()
        results[r] = results.get(r, 0) + 1
        if r == 1:
          result1 += 1
    return result1