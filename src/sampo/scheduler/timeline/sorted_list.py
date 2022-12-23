from abc import ABC
from bisect import bisect_left, bisect_right

from sortedcontainers import SortedKeyList


# TODO: describe the class (description)
class ExtendedSortedList(SortedKeyList, ABC):
    # TODO: describe the function (parameters (old), return type)
    def merge(self, old, value, merger):
        """
        Merges `value` and `old` and places the result to the place of `old` in sorted list.
        This should NOT modify key of object.

        Runtime complexity: `O(log(n))` -- approximate.

        :param merger: a function that takes old and new value and merges it into new one
        :param value: value to merge

        """
        _lists = self._lists
        _keys = self._keys
        _maxes = self._maxes

        key = self._key(old)

        increased = True

        if _maxes:
            pos = bisect_left(_maxes, key)

            # def insert_or_merge():
            #     idx = bisect_right(_keys[pos], key)
            #     old_key = _keys[pos][idx]
            #     if old_key == key:
            #         _lists[pos][idx] = merger(_lists[pos][idx], value)
            #     else:
            #         _lists[pos].insert(idx, value)
            #         _keys[pos].insert(idx, key)

            if pos == len(_maxes):
                pos -= 1

                old_key = _keys[pos][-1]
                if old_key == key:
                    new_value = merger(_lists[pos][-1], value)
                    _lists[pos][-1] = new_value
                    # _keys[pos][-1] = self._key(new_value)
                    increased = False
                else:
                    _lists[pos].append(value)
                    _keys[pos].append(key)
                _maxes[pos] = key
            else:
                idx = bisect_right(_keys[pos], key)
                old_key = _keys[pos][idx - 1]
                if idx != 0 and old_key == key:
                    new_value = merger(_lists[pos][idx - 1], value)
                    _lists[pos][idx - 1] = new_value
                    # _keys[pos][idx - 1] = self._key(new_value)
                    increased = False
                else:
                    _lists[pos].insert(idx, value)
                    _keys[pos].insert(idx, key)

            if increased:
                self._expand(pos)
        else:
            _lists.append([value])
            _keys.append([key])
            _maxes.append(key)

        if increased:
            self._len += 1
