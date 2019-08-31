from collections import defaultdict
import sys
import numpy as np
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
    tags = ['rmse_batch', 'rmse_train', 'rmse_val', 'rmse_test']
    tag_values = {k : np.array(v) for v in extract_values(sys.argv[1], tags)}
    print(eval(sys.argv[2]))
