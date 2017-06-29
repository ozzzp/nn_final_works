# -*- coding: utf-8 -*-
import argparse

from Othello.Othello_net import *

parser = argparse.ArgumentParser(description='Othello Game with MCTS and a Small Q-net')
parser.add_argument('--cuda', type=bool, default=False, metavar='True/False',
                    help='should or not to use GPU accelerate')
parser.add_argument('--mode', type=str, default='Eval', metavar='Train/Eval',
                    help='select Train mode or Eval')
parser.add_argument('--pertrain', type=str, default='V8', metavar='<model name>',
                    help='select the pertrain model')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode == 'Train':
        players = [None, '_']
        top_player = players[-1]
        decay = 0.8
        p = [decay ** i for i in range(len(players))][::-1]
        print(p)
        for i in range(len(players) - 1, 101):
            name = 'V' + str(i)
            print(name)
            train(name, top_player, players, max_iter=70000, times=9 * 11, p=p, value_policy=value_policy,
                  cuda=args.cuda, cpus=9)
            top_player = name
            players.append(top_player)
            p = [i * decay for i in p]
            p.append(1)
    elif args.mode == 'Eval':
        path = os.path.join(value_policy, args.pertrain)
        model = ValueNet()
        model.load_state_dict(torch.load(path))
        if args.cuda:
            model.cuda()
        model.eval()
        default_policy = net_play_gen(model, cuda=args.cuda, gain=12.5)
        tree_policy, prepare_tree_policy = UCT_with_net_gen(model, lam=lam, cuda=args.cuda)
        player = mcts_search_gen(default_play=default_policy,
                                 tree_policy=tree_policy,
                                 # simulation=expect_simulation_gen(model, gain=12.5),
            prepare_tree_policy=prepare_tree_policy,
                                 max_sec=4,
                                 should_print=True)
        exp = 0
        for i in range(1, 1000):
            game = composition()
            mcts = mcts_search_gen(max_sec=4)
            exp += game.run(black_play=mcts,
                            white_play=player,
                            should_print=True)
            # print(game)
            game = composition()
            exp -= game.run(black_play=player,
                            white_play=mcts,
                            should_print=False)
            # print(game)
            print(exp / 2 / i)
    else:
        raise ValueError
