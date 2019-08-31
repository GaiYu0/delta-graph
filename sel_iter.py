# python3 -W ignore select-iteration.py PATH METRIC

from collections import defaultdict
import os
import sys
from event_file_loader import EventFileLoader

def extract_values(file_path, tags):
    loader = EventFileLoader(file_path)
    tag_values = defaultdict(list)
    for event in loader.Load():
        try:
            tag = event.summary.value[0].tag
            if tag in tags:
                tag_values[tag].append(event.summary.value[0].simple_value)
        except (AttributeError, IndexError):
            pass
    return tag_values

if __name__ == '__main__':
    b = extract_values('%s/b/%s' % (sys.argv[1], os.listdir('%s/b' % sys.argv[1])[0]))
    c = extract_values('%s/c/%s' % (sys.argv[1], os.listdir('%s/c' % sys.argv[1])[0]))

    # TODO tensorboard bug?
    min_len = min(len(b), len(c))
    b = b[:min_len]
    c = c[:min_len]
    inf = float('Inf')
    predicate = lambda pair: pair[0] != inf and pair[1] != inf
    try:
        b, c = map(list, zip(*filter(predicate, zip(b, c))))
    except ValueError:
        pass

    if b:
        x = max(b)
        print('%.3f %.3f' % (x, c[b.index(x)]))
    else:
        print('0.0 0.0')
