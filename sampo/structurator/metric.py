from collections import namedtuple
from math import inf
from typing import Dict, List, Set


def is_sequence_correct(adj_matrix: Dict[str, Dict[str, List[str]]], sequence: List[str]) -> (
        bool, Dict[str, Set[str]]):
    """
    Checks the sequence for correctness and builds an adjacency matrix containing
    all hard links and soft links performed
    :param adj_matrix: adjacency matrix, where for each key the value is a dictionary {'hard': [], 'soft': []}
    :param sequence: vertex sequence from the chromosome
    :return:
        is_correct (bool): True if the sequence is correct, otherwise False
        correct_adj_matrix (Dict[str, Set[str]]): if is_correct is True, then the matrix is from the description, otherwise None
    """
    # Check that all ids are present in the sequence and exactly once
    if len(set(adj_matrix.keys())) != len(set(sequence)):
        return False, None

    # searching for uncompleted hard ties
    used: Set[str] = set()
    correct_adj_matrix: Dict[str, Set[str]] = dict()
    for s in sequence:
        used.add(s)
        hard_children = set(adj_matrix[s]['hard'])
        if len(used & hard_children) > 0:
            return False, None
        correct_adj_matrix[s] = hard_children
        soft_children = set(adj_matrix[s]['soft']) - used
        correct_adj_matrix[s] |= soft_children
    return True, correct_adj_matrix


def soft_metric_score(adj_matrix: Dict[str, Dict[str, List[str]]], sequence: List[str]) -> float:
    """
        counts the metric of the number of soft ties
        :param adj_matrix: adjacency matrix, where for each key the value is a dictionary {'hard': [], 'soft': []}
        :param sequence: vertex sequence from the chromosome
        :return:
            score (float): The number of executed soft links if the sequence is correct, otherwise math.inf
    """
    # Check that all ids are present in the sequence and exactly once
    is_correct, correct_adj = is_sequence_correct(adj_matrix, sequence)
    if not is_correct:
        return inf

    score = sum(len(correct_adj[key] - set(adj_matrix[key]['hard'])) for key in sequence)
    return score


def mean_path_score(adj_matrix: Dict[str, Dict[str, List[str]]], sequence: List[str]) -> float:
    """
        counts the metric of the mean length path
        :param adj_matrix: adjacency matrix, where for each key the value is a dictionary {'hard': [], 'soft': []}
        :param sequence: vertex sequence from the chromosome
        :return:
            score (float): The mean length path in the graph if the sequence is correct, otherwise math.inf
    """
    # Check that all ids are present in the sequence and exactly once
    is_correct, correct_adj = is_sequence_correct(adj_matrix, sequence)
    if not is_correct:
        return inf

    PathCounter = namedtuple('PathCounter', 'sum count')
    counter: Dict[str, PathCounter] = dict()

    def dfs(v_id: str):
        if v_id in counter:
            return
        if len(correct_adj[v_id]) == 0:
            counter[v_id] = PathCounter(0, 1)
            return
        path_sum, path_count = 0, 0
        for child in correct_adj[v_id]:
            dfs(child)
            child_path = counter[child]
            path_sum += child_path.sum + child_path.count
            path_count += child_path.count
        counter[v_id] = PathCounter(path_sum, path_count)
        return

    res_sum, res_count = 0, 0
    for s_id in sequence:
        if s_id not in counter:
            dfs(s_id)
            res_sum += counter[s_id].sum
            res_count += counter[s_id].count
    return res_sum / res_count


if __name__ == "__main__":
    test_adj_matrix = {
        '1': {'hard': ['2', '3', '4'], 'soft': ['5', ]},
        '2': {'hard': ['5', ], 'soft': []},
        '3': {'hard': ['11'], 'soft': []},
        '4': {'hard': ['5'], 'soft': ['3']},
        '5': {'hard': [], 'soft': []},
        '6': {'hard': ['7', '8'], 'soft': ['9']},
        '7': {'hard': ['10'], 'soft': []},
        '8': {'hard': ['9'], 'soft': []},
        '9': {'hard': ['10'], 'soft': ['7']},
        '10': {'hard': [], 'soft': []},
        '11': {'hard': ['5'], 'soft': ['2']},
        '12': {'hard': ['15'], 'soft': ['14']},
        '13': {'hard': ['15'], 'soft': ['14']},
        '14': {'hard': ['15'], 'soft': []},
        '15': {'hard': [], 'soft': []},
    }

    slice_adj = {key: test_adj_matrix[key] for key in ['12', '13', '14', '15']}
    short_tests = [
        (slice_adj, ['12', '13', '14', '15'],
         (True, {'12': {'14', '15'}, '13': {'14', '15'}, '14': {'15'}, '15': set()}), 2, 1.5),
        (slice_adj, ['12', '13', '14', '14'], (False, None), inf, inf),
        (slice_adj, ['12', '13', '15', '14'], (False, None), inf, inf),
        (slice_adj, ['14', '13', '12', '15'],
         (True, {'14': {'15'}, '13': {'15'}, '12': {'15'}, '15': set()}), 0, 1.0),
    ]
    print(any(is_sequence_correct(test[0], test[1]) == test[2] for test in short_tests))
    print(any(soft_metric_score(test[0], test[1]) == test[3] for test in short_tests))
    print(any(mean_path_score(test[0], test[1]) == test[4] for test in short_tests))
