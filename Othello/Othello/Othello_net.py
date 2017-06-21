# -*- coding: utf-8 -*-
import os
import pickle
import queue

import torch
import torch.multiprocessing as multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    multiprocessing.set_start_method('spawn')
except:
    pass
from torch.autograd import Variable

from .Othello_base import *

lam = 0.5

current_dir = os.getcwd()

log_dict = os.path.join(current_dir, 'log')
value_policy = os.path.join(current_dir, 'vp')

if not (os.path.exists(log_dict) and os.path.isdir(log_dict)):
    os.mkdir(log_dict)
if not (os.path.exists(value_policy) and os.path.isdir(value_policy)):
    os.mkdir(value_policy)


class ValueNet(nn.Module):
    def __init__(self, channels=64, depth=5):
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


def mask(candidate, cuda=False):
    mask = torch.ByteTensor(1, 1, 8, 8).zero_()
    for i in candidate:
        mask[0, 0, i[0], i[1]] = 1
    if cuda:
        return mask.cuda()
    else:
        return mask


def net_play_gen(model, cuda=False, gain=6):
    assert isinstance(model, ValueNet)

    def net_play(compos):
        actions = sorted(list(compos.current_candidate))
        current = np.array(compos.current * compos.current_target, dtype=np.float32)
        current = torch.from_numpy(current[np.newaxis, np.newaxis, :, :])
        if cuda:
            current = current.cuda()
        current = Variable(current)
        p = F.softmax(gain * model(current)[mask(compos.current_candidate, cuda=cuda)])
        if cuda:
            p = p.cpu()
        # action = np.argmax(p.data.numpy())
        action = np.random.choice(len(actions), 1, p=p.data.numpy())[0]
        action = actions[action]
        return action

    return net_play


def fixed_prb_simulation_gen(model, cuda=False, gain=6):
    assert isinstance(model, ValueNet)

    def generate_prob_play(p):
        def random_play(compos):
            actions = sorted(list(compos.current_candidate))
            prb = F.softmax(p[mask(compos.current_candidate, cuda=cuda)])
            if cuda:
                prb = p.cpu()
            action = np.random.choice(len(actions), 1, p=prb.data.numpy())[0]
            action = actions[action]
            return action

        return random_play

    def get_p(current, target=black):
        current = np.array(current * target, dtype=np.float32)
        current = torch.from_numpy(current[np.newaxis, np.newaxis, :, :])
        if cuda:
            current = current.cuda()
        current = Variable(current)
        result = model(current)[0, 0, :, :]
        p = gain * result
        return p

    def simulation(compos):
        new_compos = copy.deepcopy(compos)
        black_play = generate_prob_play(get_p(new_compos.current, target=black))
        white_play = generate_prob_play(get_p(new_compos.current, target=white))
        return new_compos.run(black_play=black_play, white_play=white_play, should_print=False)

    return simulation


def expect_simulation_gen(model, cuda=False, gain=6):
    assert isinstance(model, ValueNet)

    def simulation(compos):
        current = np.array(compos.current * compos.current_target, dtype=np.float32)
        current = torch.from_numpy(current[np.newaxis, np.newaxis, :, :])
        if cuda:
            current = current.cuda()
        current = Variable(current)
        result = model(current)
        result = result[mask(compos.current_candidate, cuda=cuda)]
        p = F.softmax(gain * result)
        expect = compos.current_target * (2 * F.sigmoid(result) - 1)
        eval = sum(p * expect)
        if cuda:
            eval = eval.cpu()
        return eval.data[0]

    return simulation


def UCT_with_net_gen(model, lam=lam, cuda=False):
    p = torch.FloatTensor(8, 8).zero_()
    one_minus_lam = 1 - lam

    def prepare_tree_policy(node):
        nonlocal p
        if hasattr(node, 'p'):
            p = node.p
        else:
            current = np.array(node.compos.current * node.compos.current_target, dtype=np.float32)
            current = torch.from_numpy(current[np.newaxis, np.newaxis, :, :])
            if cuda:
                current = current.cuda()
            current = Variable(current)
            p = F.sigmoid(model(current))
            if cuda:
                p = p.cpu()
            p = p.data.numpy()[0, 0, :, :]
            node.p = p

    def UCT_with_net(node, target=black, repositary={}):
        """
        :param node:mcts_node
        :return: float
        """
        node, action = node
        n_f = sum(repositary[id].visited for id in node.father) if len(node.father) != 0 else node.visited
        return lam * get_expect(node, target=target) + \
               one_minus_lam * p[action] + \
               Cp * math.sqrt(2 * math.log(n_f) / node.visited) \
            if node.visited != 0 \
            else float('Inf')

    return UCT_with_net, prepare_tree_policy


def change_player(player, log=None, value_policy=value_policy, cuda=False):
    if player is None:
        return random_play
    elif player == '_':
        return mcts_search_gen(ext_log=log,
                               max_iter=450,
                               should_print=True)
    elif isinstance(player, str):
        model = ValueNet()
        state_dict = torch.load(os.path.join(value_policy, player))
        model.load_state_dict(state_dict)
        if cuda:
            model.cuda()
        model.eval()
        default_policy = net_play_gen(model, cuda=cuda)
        tree_policy, prepare_tree_policy = UCT_with_net_gen(model, lam=lam, cuda=cuda)
        return mcts_search_gen(default_play=default_policy,
                               tree_policy=tree_policy,
                               prepare_tree_policy=prepare_tree_policy,
                               ext_log=log,
                               max_iter=450,
                               should_print=True)
    else:
        raise ValueError


def run_a_game(saved_path, _upper_player, _down_player, repeat=1, value_policy=value_policy, cuda=False):
    upper_log = {}
    down_log = {}
    upper_player = change_player(_upper_player, log=upper_log, value_policy=value_policy, cuda=cuda)
    down_player = change_player(_down_player, log=down_log, value_policy=value_policy, cuda=cuda)
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
    output = open(saved_path + '_' + str(expect / 2 / repeat) + '.pkl', 'wb')
    if _down_player is None or expect >= 0:
        log = upper_log
    else:
        log = down_log
    pickle.dump(log, output)
    return log


def __collect_a_game(player, top_player=None, repeat=1, log_dict=log_dict, value_policy=value_policy, queue=None,
                     cuda=False):
    player, number = player
    np.random.seed()
    path = str(number) + str(player)
    path = os.path.join(log_dict, path)
    log = run_a_game(path, top_player, player, repeat=repeat, value_policy=value_policy, cuda=cuda)
    if queue is not None:
        queue.put(log)


def collect_data(top_player, players, times=100, repeat=1, log_dict=log_dict, value_policy=value_policy, p=None,
                 queue=None, cuda=False, cpus=None):
    assert isinstance(players, list)
    if p is not None:
        assert isinstance(p, list) and len(p) == len(players)
        sum_p = sum(p)
        p = [i / sum_p for i in p]

    log_dict = os.path.join(log_dict, top_player)

    if not (os.path.exists(log_dict) and os.path.isdir(log_dict)):
        os.mkdir(log_dict)

    play_list = np.random.choice(len(players), times, replace=True, p=p)
    play_list = [(players[play_list[i]], i) for i in range(len(play_list))]

    with multiprocessing.Pool(cpus) as pool:
        pool.map(functools.partial(__collect_a_game,
                                   top_player=top_player,
                                   repeat=repeat,
                                   log_dict=log_dict,
                                   value_policy=value_policy,
                                   queue=queue,
                                   cuda=cuda), play_list)


def flip(log):
    new_log = []
    for compos, pred, pred_weight in log:
        flipped = np.random.choice([-1, 1], 3, replace=True)
        current = compos.current * compos.current_target
        current = current[::flipped[0], ::flipped[1]]
        pred = pred[::flipped[0], ::flipped[1]]
        pred_weight = pred_weight[::flipped[0], ::flipped[1]]
        if flipped[2] == -1:
            current = current.transpose()
            pred = pred.transpose()
            pred_weight = pred_weight.transpose()
        new_log.append((current, pred, pred_weight))
    return new_log


def process_data(log, cuda=False):
    log = flip(log.values())
    data = np.array([current for current, pred, w in log], dtype=np.float32)
    data = torch.from_numpy(data[:, np.newaxis, :, :])
    mask = np.array([np.uint8(pred >= 0) for current, pred, w in log])
    mask = torch.from_numpy(mask[:, np.newaxis, :, :])
    target = np.hstack([pred[np.where(pred >= 0)] for current, pred, w in log])
    target = torch.from_numpy(np.array(target, dtype=np.float32))
    weight = np.hstack([w[np.where(pred >= 0)] for current, pred, w in log])
    weight /= np.mean(weight)
    weight = torch.from_numpy(np.array(weight, dtype=np.float32))
    if cuda:
        data, mask, target, weight = data.cuda(), mask.cuda(), target.cuda(), weight.cuda()
    return Variable(data), \
           Variable(mask), \
           Variable(target), \
           Variable(weight)


def train_one_step(model, optimizer, log, cuda=False):
    data, mask, target, weight = process_data(log, cuda=cuda)
    optimizer.zero_grad()
    output = model(data)
    output = output[mask]
    output = F.sigmoid(output)
    loss = F.binary_cross_entropy(output, target, weight=weight)
    loss.backward()
    optimizer.step()
    return loss.data[0]


def train(name, top_player, players, max_iter=20000, times=100, p=None, value_policy=value_policy, cuda=False,
          cpus=None):
    log = []
    log_prb = []
    assert isinstance(top_player, str)

    manager = multiprocessing.Manager()
    log_queue = manager.Queue()
    prepare_process = multiprocessing.Process(target=collect_data,
                                              args=(top_player, players),
                                              kwargs={'times': times,
                                                      'p': p,
                                                      'queue': log_queue,
                                                      'cuda': cuda,
                                                      'cpus': cpus})
    prepare_process.start()
    model = ValueNet()
    '''
    if top_player != '_':
        state_dict = torch.load(os.path.join(value_policy, top_player))
        model.load_state_dict(state_dict)
    '''
    if cuda:
        model.cuda()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    iter = 0
    while True:
        try:
            if len(log) == 0:
                data = log_queue.get()
            else:
                data = log_queue.get_nowait()
            log.append(data)
            log_prb.append(0)
        except queue.Empty:
            if not prepare_process.is_alive():
                break
            '''
            prb =  [math.exp(i) for i in log_prb]
            sum_prb = sum(prb)
            prb = [i / sum_prb for i in prb]
            n = np.random.choice(len(log), 1, p=prb)[0]
            '''
            n = np.argmax(log_prb)
            data = log[n]
            log_prb[n] -= 1
            time.sleep(1)
        loss = train_one_step(model, optimizer, data, cuda=cuda)
        if iter % 10 == 0:
            print(iter, loss)
        iter += 1

    if iter < max_iter:
        for i in range(iter, max_iter):
            n = np.argmax(log_prb)
            data = log[n]
            log_prb[n] -= 1
            loss = train_one_step(model, optimizer, data, cuda=cuda)
            if i % 100 == 0:
                print(i, loss)

    path = os.path.join(value_policy, name)
    if cuda:
        model.cpu()
    torch.save(model.state_dict(), path)
