# -*- coding: utf-8 -*-
import copy
import functools
import math
import os
import subprocess
import time

import numpy as np

subprocess.call(['python setup.py build_ext --inplace'], shell=True, cwd=os.getcwd())

from Othello.Othello_ext import _check_rolling_over, _generate_doomed_map

block = 0
black = 1
white = -1

Cp = 0.5


def human_play(compos):
    action = input(("Black" if compos.current_target == black else "White") + " play : ")
    assert len(action) == 2
    if action[0].isalpha() and action[1].isdigit():
        x, y = action[0], action[1]
    elif action[1].isalpha() and action[0].isdigit():
        y, x = action[0], action[1]
    else:
        raise ValueError
    x = ord(x.upper()) - ord('A')
    y = int(y) - 1
    assert 0 <= x < 8
    assert 0 <= y < 8
    return y, x


def random_play(compos):
    actions = list(compos.current_candidate)
    action = np.random.choice(len(actions), 1)[0]
    action = actions[action]
    return action


def get_expect(node, target=black):
    if node.doomed is not None:
        exp = node.doomed * float('inf') if node.doomed != 0 else 0
    elif node.visited != 0:
        exp = node.expect / node.visited
    else:
        exp = 0
    return 0.5 + 0.5 * target * exp


def UCT(node, target=black, repositary={}):
    """
    :param node:mcts_node
    :return: float
    """
    node = node[0]
    expect = get_expect(node, target=target)
    n_f = sum(repositary[id].visited for id in node.father) if len(node.father) != 0 else node.visited
    return expect / node.visited + 2 * Cp * math.sqrt(2 * math.log(n_f) / node.visited) \
        if node.visited != 0 else float('Inf')


def mcts_search_gen(default_play=random_play, tree_policy=UCT, prepare_tree_policy=None, max_sec=3600, max_iter=3 ** 64,
                    ext_log=None,
                    should_print=True):
    assert hasattr(default_play, '__call__')

    def simulation(compos):
        new_compos = copy.deepcopy(compos)
        return new_compos.run(black_play=default_play, white_play=default_play, should_print=False)

    cal_id = lambda x: (tuple(x.current.reshape(-1).tolist()), x.current_target)

    def generate_doomed_map(compos):
        doomed_map = np.ones([8, 8], dtype=np.int32) * block
        return _generate_doomed_map(doomed_map, compos.current)

    class mcts_node(object):
        def __init__(self, compos, id=None, action=None):
            """
            :param compos: composition
            """
            self.compos = copy.deepcopy(compos)
            if isinstance(action, tuple):
                self.compos.add_pos(self.compos.current_target, action, should_print=False)
                self.compos.check_state(should_print=False)
            self.doomed_map = generate_doomed_map(self.compos)
            self.expect = 0.0
            self.visited = 0.0
            self.doomed = None
            self.doomed_count = 0
            if id is None:
                self.id = cal_id(self.compos)
            else:
                self.id = id
            self.father = set()
            self.child = {}

        def _doomed_one_cut(self, node, repositary):
            if self.doomed == node.compos.current_target:
                node.doomed = self.doomed
                return True
            else:
                node.doomed_count += 1
                if node.doomed_count == len(node.child):
                    child_dooms = {repositary[id].doomed for id in node.child}
                    node.doomed = max(child_dooms) if node.compos.current_target == black else min(child_dooms)
                    return True
                else:
                    return False

        def doomed_cut(self, repositary):
            for id in self.father:
                node = repositary[id]
                if node.doomed is None:
                    if self._doomed_one_cut(node, repositary):
                        node.doomed_cut(repositary)

        def backword(self, repositary, visited=None, expect=None):
            if visited is None:
                visited = self.visited
            if expect is None:
                expect = self.expect
            backed_set = copy.copy(self.father)
            blocked_set = set()
            while len(backed_set) != 0:
                iter = repositary[backed_set.pop()]
                iter.visited += visited
                iter.expect += expect
                backed_set |= iter.father - blocked_set
                blocked_set.add(iter.id)
            if self.doomed is not None:
                self.doomed_cut(repositary)

        def get_child(self, repositary):
            self.visited += 1.0
            if self.compos.final is not None:
                self.doomed = self.compos.final
            else:
                black_doomed = len(np.where(self.doomed_map == black)[0])
                if black_doomed > 32:
                    self.doomed = black
                else:
                    white_doomed = len(np.where(self.doomed_map == white)[0])
                    if white_doomed > 32:
                        self.doomed = white
            if self.doomed is None:
                self.expect += simulation(self.compos)
            else:
                self.expect += self.doomed

            doomed_dict = {}
            for action in self.compos.current_candidate:
                child = mcts_node(self.compos, action=action)
                if child.id not in repositary:
                    repositary[child.id] = child
                else:
                    child = repositary[child.id]
                    if child.doomed is not None:
                        doomed_dict[child.id] = child
                child.father.add(self.id)
                self.child[child.id] = action
            for child in doomed_dict.values():
                if child._doomed_one_cut(self, repositary):
                    break
            self.backword(repositary)
            return

    mcts_node_repositary = {}

    def tree_select(node, repositary):
        while True:
            if len(node.child) == 0:
                break
            else:
                if hasattr(prepare_tree_policy, '__call__'):
                    prepare_tree_policy(node)
                node, _ = max(
                    ((repositary[id], action) for id, action in node.child.items() if repositary[id].doomed is None),
                    key=functools.partial(tree_policy, target=node.compos.current_target,
                                          repositary=mcts_node_repositary))
        return node

    def clean_father(node, repositary):
        node.father &= repositary
        return node

    def mcts_play(compos):
        """

        :param compos:composition
        :return:
        """
        start_time = time.time()
        nonlocal mcts_node_repositary
        root_id = cal_id(compos)
        if root_id in mcts_node_repositary:
            old_repositary = mcts_node_repositary
            mcts_node_repositary = set()
            cand_set = {root_id}
            while len(cand_set) != 0:
                id = cand_set.pop()
                cand_set |= set(old_repositary[id].child.keys()) - mcts_node_repositary
                mcts_node_repositary.add(id)
            mcts_node_repositary = {id: clean_father(old_repositary[id], mcts_node_repositary) for id in
                                    mcts_node_repositary}
            mcts_node_repositary[root_id].father = set()
        else:
            mcts_node_repositary = {root_id: mcts_node(compos, id=root_id)}
        root = mcts_node_repositary[root_id]
        for i in range(max_iter):
            if (time.time() - start_time) >= max_sec or root.doomed is not None:
                break
            node = tree_select(root, mcts_node_repositary)
            node.get_child(mcts_node_repositary)

        if len(root.child) == 0:
            best_action = default_play(compos)
        else:
            best_action = max(((mcts_node_repositary[id], action) for id, action in root.child.items()),
                              key=lambda x: get_expect(x[0], target=compos.current_target))
        if isinstance(ext_log, dict):
            pred = - np.ones([8, 8], dtype=np.float)
            for id, action in root.child.items():
                score = get_expect(mcts_node_repositary[id], target=compos.current_target)
                pred[action] = max(min(score, 1), 0)
            ext_log[root_id] = (root.compos, pred)

        if should_print:
            doomed = root.doomed
            if doomed == black:
                doomed = 'X'
            elif doomed == white:
                doomed = 'O'
            elif doomed is None:
                doomed = "%.3f" % get_expect(best_action[0], target=compos.current_target)
            elif doomed == block:
                doomed = '='
            print("%d\t%s  %s  %d\t%s" % (len(np.where(compos.current != block)[0]) + 1,
                                          'X' if compos.current_target == black else 'O',
                                          translate_to_note(best_action[1]),
                                          i,
                                          doomed))
        return best_action[1]

    return mcts_play


def translate_to_note(pos):
    assert isinstance(pos, tuple) and len(pos) == 2
    x = chr(pos[1] + ord('a'))
    y = str(pos[0] + 1)
    return x + y


class composition(object):
    def __init__(self):
        self.current = np.ones([8, 8], dtype=np.int32) * block
        self.current[3, 3] = white
        self.current[4, 4] = white
        self.current[3, 4] = black
        self.current[4, 3] = black
        self.current_target = black
        self.edge = {(2, 2), (2, 3), (2, 4), (2, 5), (3, 5), (4, 5), (5, 5), (5, 4), (5, 3), (5, 2), (4, 2), (3, 2)}
        _, self.current_candidate = self.check_game_over(self.current_target)
        self.final = None

    def __deepcopy__(self, memodict={}):
        new_copy = composition.__new__(composition)
        new_copy.current = copy.copy(self.current)
        new_copy.current_target = self.current_target
        new_copy.edge = copy.copy(self.edge)
        new_copy.current_candidate = copy.copy(self.current_candidate)
        new_copy.final = self.final
        return new_copy

    def check_state(self, should_print=False):
        next_target = black + white - self.current_target
        result, detail = self.check_game_over(next_target)
        if result == 0:
            self.current_target = next_target
            self.current_candidate = detail
        elif result == 1:
            self.current_candidate = detail
        elif result == 2:
            self.current_candidate = set()
            if should_print:
                print(("Black" if detail == black else "white") + ' wins!')
            self.final = detail
        elif result == 3:
            self.current_candidate = set()
            if should_print:
                print("Draw!")
            self.final = 0
        else:
            raise ValueError
        return self.final

    def one_step(self, play, should_print=False):
        while True:
            try:
                action = play(self)
                self.add_pos(self.current_target, action)
                break
            except ValueError:
                if play is human_play:
                    print("illegal action, retry!")
                    continue
                else:
                    import traceback
                    traceback.print_exc()
                    raise ValueError("illegal action!")
        return self.check_state(should_print=should_print)

    def run(self, black_play=human_play, white_play=human_play, max_steps=64, should_print=True):
        for i in range(max_steps):
            if should_print:
                print(i + 1)
                print(self)
            if self.current_target == black:
                current_play = black_play
            elif self.current_target == white:
                current_play = white_play
            else:
                raise ValueError
            result = self.one_step(current_play, should_print=should_print)
            if result is not None:
                if should_print:
                    print(i + 1)
                    print(self)
                return result
        else:
            return None

    def add_pos(self, target, position, should_print=False):
        y, x = position
        if (y, x) in self.current_candidate:
            self.current[y, x] = target
            self._check_rolling_over(target, y, x)
            self.expand_edge(y, x)
            if should_print:
                print(self)
        else:
            raise ValueError

    def expand_edge(self, y, x):
        for y_n, x_n in [(y - 1, x), (y + 1, x),
                         (y, x - 1), (y, x + 1),
                         (y + 1, x + 1), (y - 1, x + 1),
                         (y + 1, x - 1), (y - 1, x - 1)]:
            if 0 <= x_n < 8 and 0 <= y_n < 8 and self.current[y_n, x_n] == block:
                self.edge.add((y_n, x_n))
        self.edge.remove((y, x))

    def check_game_over(self, target):
        candidate_point = {(y, x) for y, x in self.edge if self._check_rolling_over(target, y, x, dryrun=True)}
        if len(candidate_point) != 0:
            return 0, candidate_point
        else:
            oppo_target = black + white - target
            oppo_candidate_point = {(y, x) for y, x in self.edge if
                                    self._check_rolling_over(oppo_target, y, x, dryrun=True)}
            if len(oppo_candidate_point) != 0:
                return 1, oppo_candidate_point
            else:
                count = len(np.where(self.current == target)[0])
                oppo_count = len(np.where(self.current == oppo_target)[0])
                if count == oppo_count:
                    return 3, None
                else:
                    return 2, target if count > oppo_count else oppo_target

    def __str__(self):
        prin = "  a b c d e f g h \n"
        for y in range(8):
            prin += str(y + 1)
            for x in range(8):
                if self.current[y, x] == black:
                    prin += '|X'
                elif self.current[y, x] == white:
                    prin += '|O'
                else:
                    prin += '|_'
            prin += '|' + str(y + 1) + '\n'
        prin += "  a b c d e f g h \n"
        return prin

    def _check_rolling_over(self, target, y, x, dryrun=False):
        return _check_rolling_over(self.current, target, y, x, dryrun)


if __name__ == '__main__':
    expect = 0
    log = {}
    for i in range(1, 101):
        A = composition()
        expect += A.run(black_play=mcts_search_gen(max_sec=4, ext_log=log),
                        white_play=mcts_search_gen(max_sec=4, ext_log=log),
                        should_print=False)
        print(A)
        print(expect / i, i)
    pass
