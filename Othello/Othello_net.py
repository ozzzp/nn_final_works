# -*- coding: utf-8 -*-
import multiprocessing
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from Othello.Othello_base import *

lam = 0.1

current_dir = os.getcwd()

log_dict = os.path.join(current_dir, 'log')
value_policy = os.path.join(current_dir, 'vp')

if not (os.path.exists(log_dict) and os.path.isdir(log_dict)):
    os.mkdir(log_dict)
if not (os.path.exists(value_policy) and os.path.isdir(value_policy)):
    os.mkdir(value_policy)


class ValueNet(nn.Module):
    def __init__(self, channels=32, depth=4):
        super(ValueNet, self).__init__()
        layers = []
        layers.append(nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(channels))

        for i in range(depth):
            layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm2d(channels))

        layers.append(nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1))
        layers = tuple(layers)
        self.layers = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.layers(x)


def mask(candidate):
    mask = torch.ByteTensor(1, 1, 8, 8).zero_()
    for i in candidate:
        mask[0, 0, i[0], i[1]] = 1
    return mask


def net_play_gen(model):
    assert isinstance(model, ValueNet)

    def net_play(compos):
        actions = sorted(list(compos.current_candidate))
        current = np.array(compos.current * compos.current_target, dtype=np.float32)
        current = Variable(torch.from_numpy(current[np.newaxis, np.newaxis, :, :]))
        p = F.softmax(model(current)[mask(compos.current_candidate)])
        action = np.random.choice(len(actions), 1, p=p.data.numpy())[0]
        action = actions[action]
        return action

    return net_play


def UCT_with_net_gen(model, lam=lam):
    p = torch.FloatTensor(8, 8).zero_()
    one_minus_lam = 1 - lam

    def prepare_tree_policy(node):
        nonlocal p
        if hasattr(node, 'p'):
            p = node.p
        else:
            current = np.array(node.compos.current * node.compos.current_target, dtype=np.float32)
            current = Variable(torch.from_numpy(current[np.newaxis, np.newaxis, :, :]))
            p = F.sigmoid(model(current)).data.numpy()[0, 0, :, :]
            node.p = p

    def UCT_with_net(node, target=black, repositary={}):
        """
        :param node:mcts_node
        :return: float
        """
        node, action = node
        expect = get_expect(node, target=target)
        expect = lam * expect + one_minus_lam * p[action]
        n_f = sum(repositary[id].visited for id in node.father) if len(node.father) != 0 else node.visited
        return expect / node.visited + 2 * Cp * math.sqrt(2 * math.log(n_f) / node.visited) \
            if node.visited != 0 else float('Inf')

    return UCT_with_net, prepare_tree_policy


def change_player(player, log=None, value_policy=value_policy):
    if player is None:
        return random_play
    elif player == '_':
        return mcts_search_gen(default_play=random_play,
                               tree_policy=UCT,
                               prepare_tree_policy=None,
                               ext_log=log,
                               max_sec=4,
                               should_print=False)
    elif isinstance(player, str):
        model = ValueNet()
        model.load_state_dict(torch.load(os.path.join(value_policy, player)))
        model.eval()
        default_policy = net_play_gen(model)
        tree_policy, prepare_tree_policy = UCT_with_net_gen(model, lam=lam)
        return mcts_search_gen(default_play=default_policy,
                               tree_policy=tree_policy,
                               prepare_tree_policy=prepare_tree_policy,
                               ext_log=log,
                               max_sec=4,
                               should_print=False)


def run_a_game(saved_path, _upper_player, _down_player, repeat=1, value_policy=value_policy):
    log = {}
    upper_player = change_player(_upper_player, log=log, value_policy=value_policy)
    down_player = change_player(_down_player, value_policy=value_policy)
    expect = 0
    for i in range(repeat):
        compos = composition()
        expect += compos.run(black_play=upper_player,
                             white_play=down_player,
                             should_print=False)
        compos = composition()
        expect -= compos.run(black_play=down_player,
                             white_play=upper_player,
                             should_print=False)
    print(_upper_player, 'vs', _down_player, 'expect:', expect / 2 / repeat)
    output = open(saved_path, 'wb')
    pickle.dump(log, output)
    return log


def __collect_a_game(player, top_player=None, repeat=1, log_dict=log_dict, value_policy=value_policy, queue=None):
    player, number = player
    np.random.seed()
    path = str(number) + str(player) + '.pkl'
    path = os.path.join(log_dict, path)
    log = run_a_game(path, top_player, player, repeat=repeat, value_policy=value_policy)
    if queue is not None:
        queue.put(log)


def collect_data(top_player, players, times=100, repeat=1, log_dict=log_dict, value_policy=value_policy, p=None,
                 queue=None):
    assert isinstance(players, list)
    if p is not None:
        assert isinstance(p, list) and len(p) == len(players)
        sum_p = sum(p)
        p = [i / sum_p for i in p]

    log_dict = os.path.join(log_dict, str(top_player))

    if not (os.path.exists(log_dict) and os.path.isdir(log_dict)):
        os.mkdir(log_dict)

    play_list = np.random.choice(len(players), times, replace=True, p=p)
    play_list = [(players[play_list[i]], i) for i in range(len(play_list))]

    with multiprocessing.Pool() as pool:
        pool.map(functools.partial(__collect_a_game,
                                   top_player=top_player,
                                   repeat=repeat,
                                   log_dict=log_dict,
                                   value_policy=value_policy,
                                   queue=queue), play_list)


def process_data(log, cuda=False):
    log = list(log.values())
    data = np.array([compos.current * compos.current_target for compos, pred in log], dtype=np.float32)
    data = torch.from_numpy(data[:, np.newaxis, :, :])
    mask = np.array([np.uint8(pred >= 0) for compos, pred in log])
    mask = torch.from_numpy(mask[:, np.newaxis, :, :])
    target = np.hstack([pred[np.where(pred >= 0)] for compos, pred in log])
    target = torch.from_numpy(np.array(target, dtype=np.float32))
    if cuda:
        data, mask, target = data.cuda(), mask.cuda(), target.cuda()
    return Variable(data), \
           Variable(mask), \
           Variable(target)


def train_one_step(model, optimizer, log, cuda=False):
    data, mask, target = process_data(log, cuda=cuda)
    optimizer.zero_grad()
    output = model(data)
    output = output[mask]
    output = F.sigmoid(output)
    loss = F.binary_cross_entropy(output, target)
    print("%.4f" % loss.data[0])
    loss.backward()
    optimizer.step()


def train(name, top_player, players, times=100, p=None, value_policy=value_policy, cuda=False):
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    prepare_process = multiprocessing.Process(target=collect_data,
                                              args=(top_player, players),
                                              kwargs={'times': times, 'p': p, 'queue': queue})
    prepare_process.start()
    model = ValueNet()
    if top_player != '_':
        model.load_state_dict(torch.load(os.path.join(value_policy, top_player)))
    if cuda:
        model.cuda()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    while prepare_process.is_alive() or not queue.empty():
        data = queue.get()
        train_one_step(model, optimizer, data, cuda=True)

    path = os.path.join(value_policy, name)
    torch.save(model.state_dict(), path)


if __name__ == '__main__':
    top_player = '_'
    players = [None, '_']
    decay = 2 / 3
    p = [decay, 1]

    for i in range(1, 11):
        name = 'V' + str(i)
        print(name)
        train(name, top_player, players, times=4, p=p, value_policy=value_policy)
        top_player = name
        players.append(top_player)
        p = [i * decay for i in p]
        p.append(1)
    pass
