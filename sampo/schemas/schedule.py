from datetime import datetime
from functools import partial, lru_cache
from typing import Tuple, List, Dict, Iterable, Union, Set

from pandas import DataFrame

from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.scheduled_work import ScheduledWork
from sampo.schemas.serializable import JSONSerializable, T
from sampo.schemas.time import Time
from sampo.schemas.works import WorkUnit
from sampo.utilities.datetime import add_time_delta
from sampo.utilities.schedule import fix_split_tasks

ResourceSchedule = Dict[str, List[Tuple[Time, Time]]]
ScheduleWorkDict = Dict[str, ScheduledWork]


# TODO: Rebase object onto ScheduleWorkDict and ordered ScheduledWork list
class Schedule(JSONSerializable['Schedule']):
    """
    Represents work schedule. Is a wrapper around DataFrame with specific structure.
    """
    _schedule: DataFrame

    _data_columns: List[str] = ['idx', 'task_id', 'task_name', 'contractor', 'cost',
                                'volume', 'measurement', 'successors', 'start',
                                'finish', 'duration', 'workers']
    _scheduled_work_column: str = 'scheduled_work_object'

    _columns: List[str] = _data_columns + [_scheduled_work_column]

    @property
    def full_schedule_df(self) -> DataFrame:
        """
        The full schedule DataFrame with all works, data columns and a distinct column for ScheduledWork objects

        :return: Full schedule DataFrame.
        """
        return self._schedule

    @property
    def pure_schedule_df(self) -> DataFrame:
        """
        Schedule DataFrame without service units and containing only original columns (stored in _data_columns field).

        :return: Pure schedule DataFrame.
        """
        return self._schedule[~self._schedule.apply(
            lambda row: row[self._scheduled_work_column].work_unit.is_service_unit,
            axis=1
        )][self._data_columns]

    @property
    def works(self) -> Iterable[ScheduledWork]:
        """
        Enumerates ScheduledWorks in the Schedule.

        :return: Iterable collection of all the scheduled works.
        """
        return self._schedule.scheduled_work_object

    @property
    def to_schedule_work_dict(self) -> ScheduleWorkDict:
        """
        Builds a ScheduleWorkDict from the Schedule.

        :return: ScheduleWorkDict with all the scheduled works.
        """
        return {r['task_id']: r['scheduled_work_object'] for _, r in self._schedule.iterrows()}

    @property
    def execution_time(self) -> Time:
        """
        Calculates total schedule execution time.

        :return: Finish time of the last work.
        """
        return Time(self._schedule.iloc[-1].finish)

    def __init__(self, schedule: DataFrame) -> None:
        """
        Initializes new `Schedule` object as a wrapper around `DataFrame` with specific structure.
        Don't use manually. Create Schedule `objects` via `from_scheduled_works` factory method.

        :param schedule: Prepared schedule `DataFrame`.
        """
        self._schedule = schedule

    # [SECTION] JSONSerializable overrides
    def _serialize(self) -> T:
        # Method described in base class
        return {
            'works': [sw._serialize() for sw in self._schedule.scheduled_work_object]
        }

    @classmethod
    def _deserialize(cls, dict_representation: T) -> 'Schedule':
        # Method described in base class
        dict_representation['works'] = [ScheduledWork._deserialize(sw) for sw in dict_representation['works']]
        return Schedule.from_scheduled_works(**dict_representation)

    @lru_cache
    def merged_stages_datetime_df(self, offset: Union[datetime, str]) -> DataFrame:
        """
        Merges split stages of same works after lag optimization and returns schedule DataFrame shifted to start.
        :param offset: Start of schedule, to add as an offset.
        :return: Shifted schedule DataFrame with merged tasks.
        """
        result = fix_split_tasks(self.offset_schedule(offset))
        return result

    def offset_schedule(self, offset: Union[datetime, str]) -> DataFrame:
        """
        Returns full schedule object with `start` and `finish` columns pushed by date in `offset` argument.
        :param offset: Start of schedule, to add as an offset.
        :return: Shifted schedule DataFrame.
        """
        r = self._schedule.loc[:, :]
        r['start_offset'] = r['start'].apply(partial(add_time_delta, offset))
        r['finish_offset'] = r['finish'].apply(partial(add_time_delta, offset))
        r = r.rename({'start': 'start_', 'finish': 'finish_',
                      'start_offset': 'start', 'finish_offset': 'finish'}, axis=1) \
            .drop(['start_', 'finish_'], axis=1)
        return r

    @staticmethod
    def from_scheduled_works(works: Iterable[ScheduledWork],
                             wg: WorkGraph = None) \
            -> 'Schedule':
        """
        Factory method to create a Schedule object from list of Schedule works and additional info
        :param wg: Work graph.
        :param works: Iterable collection of ScheduledWork's.
        :return: Schedule.
        """
        ordered_task_ids = order_nodes_by_start_time(works, wg) if wg else None

        def info(work_unit: WorkUnit) -> Tuple[float, str, List[Tuple[str, str]]]:
            if wg is None:
                return 0, "", []
            # noinspection PyTypeChecker
            return work_unit.volume, work_unit.volume_type, \
                   [(edge.finish.id, edge.type.value) for edge in wg[work_unit.id].edges_from]

        def sed(t1, t2) -> tuple:
            """
            Sorts times and calculates difference.
            :param t1: time 1.
            :param t2: time 2.
            :return: Tuple: start, end, duration.
            """
            s, e = tuple(sorted((t1, t2)))
            return s, e, e - s

        df = [(i,                                                 # idx
               w.work_unit.id,                                    # task_id
               w.work_unit.name,                                  # task_name
               w.contractor,                                      # contractor info
               w.cost,                                            # work cost
               *info(w.work_unit),                                # volume, measurement, successors
               *sed(*(t.value for t in w.start_end_time)),        # start, end, duration
               repr(dict((i.name, i.count) for i in w.workers)),  # workers
               w  # full ScheduledWork info
               ) for i, w in enumerate(works)]
        df = DataFrame.from_records(df, columns=Schedule._columns)

        df = df.set_index('idx')

        if ordered_task_ids:
            df.task_id = df.task_id.astype('category')
            df.task_id = df.task_id.cat.set_categories(ordered_task_ids)
            df = df.sort_values(['task_id'])
            df.task_id = df.task_id.astype(str)

        df = df.reindex(columns=Schedule._columns)
        df = df.reset_index(drop=True)

        return Schedule(df)


def order_nodes_by_start_time(works: Iterable[ScheduledWork], wg: WorkGraph) -> List[str]:
    """
    Makes ScheduledWorks' ordering that satisfies:
    1. Ascending order by start time
    2. Toposort

    :param works:
    :param wg:
    :return:
    """
    res = []
    order_by_start_time = [(item.start_time, item.work_unit.id) for item in
                           sorted(works, key=lambda item: item.start_time)]

    cur_time = 0
    cur_class: Set[GraphNode] = set()
    for start_time, work in order_by_start_time:
        node = wg[work]
        if len(cur_class) == 0:
            cur_time = start_time
        if start_time == cur_time:
            cur_class.add(node)
            continue
        # TODO Perform real toposort
        cur_not_added: Set[GraphNode] = set(cur_class)
        while len(cur_not_added) > 0:
            for cur_node in cur_class:
                if any([parent_node in cur_not_added for parent_node in cur_node.parents]):
                    continue  # we add this node later
                res.append(cur_node.id)
                cur_not_added.remove(cur_node)
            cur_class = set(cur_not_added)
        cur_time = start_time
        cur_class = {node}

    cur_not_added: Set[GraphNode] = set(cur_class)
    while len(cur_not_added) > 0:
        for cur_node in cur_class:
            if any([parent_node in cur_not_added for parent_node in cur_node.parents]):
                continue  # we add this node later
            res.append(cur_node.id)
            cur_not_added.remove(cur_node)
        cur_class = set(cur_not_added)

    return res
