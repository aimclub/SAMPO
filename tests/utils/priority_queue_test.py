from random import Random

from utilities.priority_queue import PriorityQueue


def test_extract_max():
    q = PriorityQueue.empty()
    for i in range(10000):
        q.add(i)

    for i in range(10000 - 1, 0, -1):
        assert q.extract_extremum() == i


def test_extract_min():
    q = PriorityQueue.empty(descending=True)
    for i in range(10000):
        q.add(i)

    for i in range(10000):
        assert q.extract_extremum() == i


def test_out_of_order_min():
    q = PriorityQueue.empty(descending=True)
    rand = Random()
    r = [i for i in range(10000)]
    rand.shuffle(r)
    for i in r:
        q.add(i)

    for i in range(10000):
        assert q.extract_extremum() == i


def test_out_of_order_max():
    q = PriorityQueue.empty()
    rand = Random()
    r = [i for i in range(10000)]
    rand.shuffle(r)
    for i in r:
        q.add(i)

    for i in range(10000 - 1, 0, -1):
        assert q.extract_extremum() == i
