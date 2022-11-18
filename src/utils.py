import numpy as np

__all__ = ['ProgressBar','sort_rows','get_rank']

class ProgressBar:
    def __init__(self, total=1, nchar=50, char='=', arrow='>',
                 template='|{}{}|[{: >3d}%]'):
        self.total = total
        self.nchar = nchar
        self.char = char
        self.arrow = arrow
        self.template = template
        self.pre_length = 0
        self.pre_p = -1

    def goto(self, p):
        p /= self.total
        assert 0 <= p <= 1
        percent = int(100 * p)
        if percent == self.pre_p:
            return
        self.pre_p = percent
        chars = self.char * int(p * self.nchar)
        if len(chars) == self.nchar:
            part2 = ''
        else:
            part2 = self.arrow + ' ' * (self.nchar - len(chars) - 1)
        print_string = self.template.format(chars, part2, percent)
        print('\b' * self.pre_length, end='')
        print(print_string, end='', flush=True)
        self.pre_length = len(print_string)
        if p == 1:
            print()

    def quit(self):
        self.goto(self.total)

def sort_rows(data, keyList, stable=True):
    """Sort data (2d-array) by multi-columns, return change index.

    Notes
    -----
    1. Stable sorting.
    2. In order to sort the first column both ascending and descending,
the keyList should be 1-indexed.
    3. If data is numeric, multiple -1 is a better choice.

    Parameters
    ----------
    data    : Array to be sorted.

    keyList : sequence of integer
        The sequence of columns to sort the data. The negative value
means descending. For instance, [2, -1] means the data should be sorted
by the second column ascending, then by the first column descending.

    """
    keyList = np.asarray(keyList)[::-1]
    if np.any(keyList == 0):
        raise Exception('keyList should be 1-indexed!')

    data = data[:, abs(keyList) - 1].T
    if np.all(keyList > 0):
        return np.lexsort(data)

    return np.lexsort([row if k > 0 else -get_rank(row, 'dense')
                       for k, row in zip(keyList, data)])

def get_rank(var, method='dense'):
    """Assign ranks to data, dealing with ties appropriately.

    Ranks begin at 1.

    Parameters
    ----------
    method: {'min', 'max', 'dense', 'ordinal', 'average'}

    mostly from scipy
    """
    if method not in {'min', 'max', 'dense', 'ordinal', 'average'}:
        raise ValueError('Unknown method "{}"'.format(method))

    var = np.asarray(var)
    kind = 'mergesort' if method == 'ordinal' else 'quicksort'
    idx = np.argsort(var, kind=kind)
    n = idx.size
    rank = np.empty(n, dtype=int)
    rank[idx] = np.arange(n)

    if method == 'ordinal':
        return rank + 1

    var = var[idx]
    index = np.concatenate(([True], var[:-1] != var[1:], [True]))
    dense = np.cumsum(index)[rank]

    if method == 'dense':
        return dense

    count = np.nonzero(index)[0]

    if method == 'min':
        return count[dense - 1] + 1

    if method == 'max':
        return count[dense]

    if method == 'average':
        return .5 * (count[dense - 1] + 1 + count[dense])
