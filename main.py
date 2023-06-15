from policy.puct import PUCTPolicy
from simulator import TicTacToeSimulator

def main():
    # 学習済みモデルのパス
    model_path = './model/trained_model.pth'  # 保存したモデルのパスを指定

    # 学習済みモデルの読み込み
    trained_policy = PUCTPolicy()
    trained_policy.load_model(model_path)

    # ゲームの初期化
    simulator = TicTacToeSimulator()

    while not simulator.is_game_over():
        current_player = simulator.get_current_player()

        if current_player == 'X':
            # 人間の入力を受け取る
            action = get_human_input(simulator)
        else:
            # 学習済みAIのアクションを選択
            action = trained_policy.select_action(simulator)

        # アクションを適用
        simulator.apply_action(action)

        # ゲームの状態を表示
        print(simulator.get_board_state())

    # ゲームの結果を表示
    winner = simulator.get_winner()
    if winner is None:
        print("引き分け")
    else:
        print(f"{winner}の勝利")

def get_human_input(simulator):
    while True:
        try:
            action = int(input("次の手を入力してください (0-8): "))
            if 0 <= action <= 8 and simulator.board[action] == '-':
                return action
            else:
                print("無効な手です。再入力してください。")
        except ValueError:
            print("無効な入力です。再入力してください。")

if __name__ == '__main__':
    main()