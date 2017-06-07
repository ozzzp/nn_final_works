# -*- coding: utf-8 -*-
import copy
import functools
import math
import time

import numpy as np

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


def mcts_search_gen(default_play=random_play, max_sec=3600, max_iter=3 ** 64):
    assert hasattr(default_play, '__call__')

    def simulation(compos):
        new_compos = copy.deepcopy(compos)
        return new_compos.run(black_play=default_play, white_play=default_play, should_print=False)

    cal_id = lambda x: (tuple(x.current.reshape(-1).tolist()), x.current_target)

    class mcts_node:
        def __init__(self, compos, action=None, id=None):
            """
            :param compos: composition
            """
            self.compos = copy.deepcopy(compos)
            if action is not None:
                self.visited = 1.0
                self.compos.add_pos(self.compos.current_target, action, should_print=False)
                result = self.compos.check_state(should_print=False)
                if result is None:
                    self.expect = simulation(self.compos)
                    self.doomed = None
                else:
                    self.expect = result
                    self.doomed = result
            else:
                self.expect = 0.0
                self.visited = 0.0
                self.doomed = None
            if id is None:
                self.id = cal_id(self.compos)
            else:
                self.id = id
            self.father = set()
            self.child = {}

        def doomed_cut(self, repositary):
            for id in self.father:
                node = repositary[id]
                if self.doomed == node.compos.current_target:
                    node = repositary[id]
                    node.doomed = self.doomed
                    node.compos.current_candidate = set()
                    node.doomed_cut(repositary)
                elif len(node.compos.current_candidate) == 0:
                    child_dooms = {repositary[id].doomed for id in node.child}
                    if None not in child_dooms:
                        node.doomed = max(child_dooms) if node.compos.current_target == black else min(child_dooms)
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
            try:
                action = default_play(self.compos)
            except:
                return None
            else:
                child = mcts_node(self.compos, action=action)
                visited = child.visited
                expect = child.expect
                if child.id not in repositary:
                    repositary[child.id] = child
                else:
                    child = repositary[child.id]
                    child.expect += expert
                    child.visited += visited
                self.compos.current_candidate.remove(action)
                child.father.add(self.id)
                self.child[child.id] = action
                child.backword(repositary, visited=visited, expect=expect)
                return child

    mcts_node_repositary = {}

    def tree_policy(node, target=black):
        """
        :param node:mcts_node
        :return: float
        """
        expert = 0.5 + 0.5 * target * node.expect
        n_f = sum(mcts_node_repositary[id].visited for id in node.father) if len(node.father) != 0 else node.visited
        return expert / node.visited + 2 * Cp * math.sqrt(2 * math.log(n_f) / node.visited) \
            if node.visited != 0 else float('Inf')

    def output_policy(node, target=black):
        node = node[0]
        return 0.5 + 0.5 * target * node.expect if node.doomed is None else node.doomed * target * 2

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
        for i in range(max_iter):
            if (time.time() - start_time) >= max_sec or mcts_node_repositary[root_id].doomed is not None:
                break
            node = max((node for node in mcts_node_repositary.values() if len(node.compos.current_candidate) != 0),
                       key=functools.partial(tree_policy, target=compos.current_target))
            node.get_child(mcts_node_repositary)

        best_action = max(
            ((mcts_node_repositary[id], action) for id, action in mcts_node_repositary[root_id].child.items()),
            key=functools.partial(output_policy, target=compos.current_target))
        doomed = mcts_node_repositary[root_id].doomed
        if doomed == black:
            doomed = 'X'
        elif doomed == white:
            doomed = 'O'
        elif doomed is None:
            doomed = '_'
        elif doomed == block:
            doomed = '='
        print(best_action[1], i, doomed)
        return best_action[1]

    return mcts_play


class composition(object):
    def __init__(self):
        self.current = np.ones([8, 8]) * block
        self.current[3, 3] = black
        self.current[4, 4] = black
        self.current[3, 4] = white
        self.current[4, 3] = white
        self.current_target = black
        self.edge = {(2, 2), (2, 3), (2, 4), (2, 5), (3, 5), (4, 5), (5, 5), (5, 4), (5, 3), (5, 2), (4, 2), (3, 2)}
        _, self.current_candidate = self.check_game_over(self.current_target)

    def check_state(self, should_print=False):
        next_target = black + white - self.current_target
        result, detail = self.check_game_over(next_target)
        if result == 0:
            self.current_target = next_target
            self.current_candidate = detail
            return None
        elif result == 1:
            self.current_candidate = detail
            return None
        elif result == 2:
            self.current_candidate = set()
            if should_print:
                print(("Black" if detail == black else "white") + ' wins!')
            return detail
        elif result == 3:
            self.current_candidate = set()
            if should_print:
                print("Draw!")
            return 0.0

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
        assert target == black or target == white
        y, x = position
        if (y, x) in self.current_candidate:
            self.current[y, x] = target
            self._check_rolling_over(target, y, x)
            self.expand_edge(y, x)
        else:
            raise ValueError
        if should_print:
            print(self)

    def expand_edge(self, y, x):
        for y_n, x_n in {(y - 1, x), (y + 1, x),
                         (y, x - 1), (y, x + 1),
                         (y + 1, x + 1), (y - 1, x + 1),
                         (y + 1, x - 1), (y - 1, x - 1)}:
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
        changed = False

        for y_p in [-1, 0, 1]:
            for x_p in [-1, 0, 1]:
                if (y_p, x_p) == (0, 0):
                    continue
                search_list = [i for i in a_line(y, x, y_p, x_p)]
                idx = 0
                should_changed = False
                for base in search_list:
                    if self.current[base] == block:
                        should_changed = False
                        break
                    elif self.current[base] == target:
                        if idx == 0:
                            should_changed = False
                        else:
                            should_changed = True
                        break
                    else:
                        idx += 1
                if not dryrun and should_changed:
                    self.current[
                        [search_list[i][0] for i in range(idx)], [search_list[i][1] for i in range(idx)]] = target
                changed |= should_changed
        return changed


def a_line(y, x, y_p, x_p):
    while 1:
        x = x + x_p
        y = y + y_p
        if 0 <= x < 8 and 0 <= y < 8:
            yield (y, x)
        else:
            break


if __name__ == '__main__':
    try:
        expert = 0
        for i in range(1, 101):
            A = composition()
            expert += A.run(black_play=random_play, white_play=mcts_search_gen(max_sec=4), should_print=False)
            print(A)
            print(expert / i)
        A = expert / i

        expert = 0
        for i in range(1, 101):
            A = composition()
            expert += A.run(black_play=mcts_search_gen(max_sec=4), white_play=random_play, should_print=False)
            print(A)
            print(expert / i)
        B = expert / i

        expert = 0
        for i in range(1, 101):
            A = composition()
            expert += A.run(black_play=mcts_search_gen(max_sec=4), white_play=mcts_search_gen(max_sec=4),
                            should_print=False)
            print(A)
            print(expert / i)
        C = expert / i
    except:
        print(A)
        print(B)
        print(C)
    else:
        print(A, B, C)
    pass
