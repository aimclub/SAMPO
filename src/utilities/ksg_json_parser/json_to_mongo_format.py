import hashlib
import json
import os
from collections import namedtuple, Counter
from os.path import exists
from typing import List, Optional, Dict, Any

FILE_UNION_ACTIVITIES = 'new_port_activities_union.json'
FILE_UNION_PROJECTS = 'new_port_projects_union.json'

ProjectType = Dict[str, Any]
ActivityType = Dict[str, Any]
UnionKey = namedtuple('UnionKey', 'name id')

UNION_KEYS = [
    UnionKey('project_codes', 'project_code_id'),
    UnionKey('global_resources', 'resource_id'),
    UnionKey('global_resource_rates', 'rate_id'),
    UnionKey('calendar', 'calendar_id'),
    UnionKey('activity_code_types', 'activity_code_type_id'),
    UnionKey('activity_codes', 'activity_code_id'),
    UnionKey('projects', 'project_id')
]


def read_and_union_jsons(path: str, processed_activity_files: Optional[List[str]] = None) \
        -> (List[ProjectType], List[ActivityType], List[str]):
    processed_activity_files = set(processed_activity_files or [])
    files_activities = sorted([file for file in os.listdir(path)
                               if "_activities.json" in file and "_union" not in file])
    files_project = sorted([file for file in os.listdir(path)
                            if "_info.json" in file and "_union" not in file])

    activities: List[ActivityType] = []
    projects_dict: Dict[str, ProjectType] = dict()

    for i in range(len(files_activities)):
        try:
            with open(path + files_activities[i]) as json_file:
                act_data = json.load(json_file)

            with open(path + files_project[i]) as json_file:
                project_data = json.load(json_file)
        except Exception:
            print(files_activities[i], " ", files_project[i])
            continue

        project = project_data['ksg_info']

        name = project["field_name"]
        if name not in projects_dict:
            if 'field_id' not in project:
                project['field_id'] = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % (10 ** 6)
            projects_dict[name] = project
        else:
            for u_key in UNION_KEYS:
                ids_set = set([elem[u_key.id] for elem in projects_dict[name][u_key.name]])
                projects_dict[name][u_key.name] += [elem for elem in project[u_key.name]
                                                    if elem[u_key.id] not in ids_set]
        for ind, _ in enumerate(act_data['activities']):
            if 'field_id' not in act_data['activities'][ind]:
                act_data['activities'][ind]['field_id'] = projects_dict[name]['field_id']
        if files_activities[i] not in processed_activity_files:
            activities += act_data['activities']

        with open(path + files_activities[i], mode='w') as json_file:
            json.dump(act_data, json_file)

        project_data['ksg_info']['field_id'] = projects_dict[name]['field_id']
        project_data['ksg_info']['field_name'] = projects_dict[name]['field_name']
        with open(path + files_project[i], mode='w') as json_file:
            json.dump(project_data, json_file)

    projects: List[ProjectType] = list(projects_dict.values())
    processed_activity_files |= set(files_activities)
    return projects, activities, list(processed_activity_files)


def write_data_to_mongo_format(path: str, name: str, jsons: List[ProjectType or ActivityType]):
    with open(path + name, 'w') as outfile:
        outfile.write(json.dumps(jsons))


if __name__ == "__main__":
    data_path = "../../../../resources/ksg_data/"
    processed_act_files = []
    if exists(data_path + 'processed_activities_files.txt'):
        with open(data_path + 'processed_activities_files.txt', 'r') as f:
            processed_act_files = [line.replace("\n", '') for line in f]
    processed_act_files = []
    projects, activities, processed_act_files = read_and_union_jsons(data_path, processed_act_files)
    print(len(projects), len(activities))
    act_counts = Counter([act['field_id'] for act in activities])
    p_counts = {(p['field_name'], p['field_id']): act_counts[p['field_id']] for p in projects}
    print(p_counts)
    write_data_to_mongo_format(data_path, FILE_UNION_PROJECTS, projects)
    write_data_to_mongo_format(data_path, FILE_UNION_ACTIVITIES, activities)
    with open(data_path + 'processed_activities_files.txt', 'w') as f:
        f.write('\n'.join(sorted(processed_act_files)))
