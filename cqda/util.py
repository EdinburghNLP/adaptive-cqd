# -*- coding: utf-8 -*-

import numpy as np
import random
import torch
import time

from typing import Dict, Tuple


def list2tuple(l):
    return tuple(list2tuple(x) if type(x)==list else x for x in l)


def tuple2list(t):
    return list(tuple2list(x) if type(x)==tuple else x for x in t)


flatten=lambda l: sum(map(flatten, l),[]) if isinstance(l,tuple) else [l]


def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())


def set_global_seed(seed: int, is_deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if is_deterministic is True:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return


def eval_tuple(arg_return):
    """Evaluate a tuple string into a tuple."""
    if type(arg_return) == tuple:
        return arg_return
    if arg_return[0] not in ["(", "["]:
        arg_return = eval(arg_return)
    else:
        splitted = arg_return[1:-1].split(",")
        List = []
        for item in splitted:
            try:
                item = eval(item)
            except:
                pass
            if item == "":
                continue
            List.append(item)
        arg_return = tuple(List)
    return arg_return


def flatten_query(queries):
    all_queries = []
    for query_structure in queries:
        tmp_queries = list(queries[query_structure])
        all_queries.extend([(query, query_structure) for query in tmp_queries])
    return all_queries


def get_query_name_dict() -> Dict[Tuple, str]:
    query_name_dict = {
        ('e', ('r',)): '1p',
        ('e', ('r', 'r')): '2p',
        ('e', ('r', 'r', 'r')): '3p',
        (('e', ('r',)), ('e', ('r',))): '2i',
        (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
        ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
        (('e', ('r', 'r')), ('e', ('r',))): 'pi',
        (('e', ('r',)), ('e', ('r', 'n'))): '2in',
        (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
        ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
        (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
        (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
        (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
        ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
        ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
        ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
    }
    return query_name_dict
