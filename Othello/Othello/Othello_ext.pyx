
cdef int block = 0

cdef inline bint in_the_line(int y, int x):
    return 0 <= x < 8 and 0 <= y < 8

def _check_rolling_over(int[:, :] current, int target, int y, int x, bint dryrun):
    cdef int y_p, x_p, idx, pos
    cdef bint should_changed
    cdef bint changed = False

    for y_p in range(-1, 2):
        for x_p in range(-1, 2):
            if (y_p, x_p) == (0, 0):
                continue
            idx = 1
            should_changed = False
            for i in range(1, 8):
                y_i = y_p * i + y
                x_i = x_p * i + x
                if not in_the_line(y_i, x_i):
                    break
                pos = current[y_i, x_i]
                if  pos == block:
                    should_changed = False
                    break
                elif pos == target:
                    if idx == 1:
                        should_changed = False
                    else:
                        should_changed = True
                    break
                else:
                    idx += 1
            if should_changed:
                if dryrun:
                    return True
                else:
                    for i in range(1, idx):
                        y_i = y_p * i + y
                        x_i = x_p * i + x
                        current[y_i, x_i] = target
                    changed = True

    return changed

def _generate_doomed_map(int[:, :] doomed_map, int[:, :] current):
    cdef int y_p, y_s, y_e, y
    cdef int x_p, x_s, x_e, x
    cdef int target

    for y_p in range(-1, 2, 2):
        if y_p == -1:
            (y_s, y_e) = (7, -1)
        else:
            (y_s, y_e) = (0, 8)
        for x_p in range(-1, 2, 2):
            if x_p == -1:
                (x_s, x_e) = (7, -1)
            else:
                (x_s, x_e) = (0, 8)
            for y in range(y_s, y_e, y_p):
                for x in range(x_s, x_e, x_p):
                    target = current[y, x]
                    if target == block:
                        break
                    elif (not 0 <= y - y_p < 8 or doomed_map[y - y_p, x] == target) \
                            and (not 0 <= x - x_p < 8 or doomed_map[y, x - x_p] == target):
                        doomed_map[y, x] = target
                    else:
                        break
                else:
                    if x == x_s:
                        break
    return doomed_map
