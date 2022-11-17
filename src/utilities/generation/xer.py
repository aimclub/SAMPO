import ast
import warnings
from typing import List

import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

XER_STRUCTURE: List[str] = ['task_id', 'proj_id', 'wbs_id', 'clndr_id', 'phys_complete_pct', 'rev_fdbk_flag', 'est_wt',
                            'lock_plan_flag', 'auto_compute_act_flag', 'complete_pct_type', 'task_type',
                            'duration_type', 'status_code', 'task_code', 'task_name', 'rsrc_id', 'total_float_hr_cnt',
                            'free_float_hr_cnt', 'remain_drtn_hr_cnt', 'act_work_qty', 'remain_work_qty',
                            'target_work_qty', 'target_drtn_hr_cnt', 'target_equip_qty', 'act_equip_qty',
                            'remain_equip_qty', 'cstr_date', 'act_start_date', 'act_end_date', 'late_start_date',
                            'late_end_date', 'expect_end_date', 'early_start_date', 'early_end_date', 'restart_date',
                            'reend_date', 'target_start_date', 'target_end_date', 'rem_late_start_date',
                            'rem_late_end_date', 'cstr_type', 'priority_type', 'suspend_date', 'resume_date',
                            'float_path', 'float_path_order', 'guid', 'tmpl_guid', 'cstr_date2', 'cstr_type2',
                            'driving_path_flag', 'act_this_per_work_qty', 'act_this_per_equip_qty',
                            'external_early_start_date', 'external_late_end_date', 'create_date', 'update_date',
                            'create_user', 'update_user', 'location_id']


def prepare_tasks_connection(task_id, pred_task_id, conn_type):
    connection_data = {'task_id': task_id, 'pred_task_id': pred_task_id, 'pred_type': 'PR_' + conn_type}
    return connection_data


def get_task_rsrc_str(ex_res_data, series_name, taskrsrc_id, task_id, task_start, task_end, res_id, res_vol,
                      create_date, update_date, create_user, update_user):
    task_data = ex_res_data.copy()
    task_data['taskrsrc_id'] = taskrsrc_id
    task_data['task_id'] = task_id
    task_data['rsrc_id'] = res_id
    task_data['guid'] = str(task_data['guid']) + str(taskrsrc_id)

    task_data['target_qty'] = res_vol
    task_data['act_reg_qty'] = res_vol
    task_data['target_cost'] = res_vol
    task_data['act_reg_cost'] = res_vol
    task_data['act_this_per_cost'] = res_vol
    task_data['act_this_per_qty'] = res_vol

    task_data['act_start_date'] = task_start
    task_data['act_end_date'] = task_end
    task_data['target_start_date'] = task_start
    task_data['target_end_date'] = task_end

    task_data['create_date'] = create_date
    task_data['update_date'] = update_date
    task_data['create_user'] = create_user
    task_data['update_user'] = update_user

    return pd.Series(data=task_data, name=series_name)


def prepare_xer_data(path_to_schedule, path_to_save, base_xer_file):
    # Global variables
    project_id = 1230
    project_name = path_to_schedule[:-4]

    electro_df = pd.read_csv(path_to_schedule, sep=';')
    electro_df = electro_df[~electro_df.task_name.str.contains('finish')]
    electro_df = electro_df[~electro_df.task_name.str.contains('start')]

    electro_df = electro_df.reset_index(drop=True)

    electro_df['workers'] = [ast.literal_eval(x) for x in electro_df.workers]
    electro_df['successors'] = [ast.literal_eval(x) for x in electro_df.successors]

    proj_start_date = str(min(electro_df.loc[:, 'start'])).split('.')[0][:-3]
    proj_finish_date = str(max(electro_df.loc[:, 'finish'])).split('.')[0][:-3]

    # Create DataFrame with the structure of main project info in XER
    proj_df = pd.DataFrame(columns=['proj_id', 'fy_start_month_num', 'rsrc_self_add_flag', 'allow_complete_flag',
                                    'rsrc_multi_assign_flag', 'checkout_flag', 'project_flag', 'step_complete_flag',
                                    'cost_qty_recalc_flag', 'batch_sum_flag', 'name_sep_char', 'def_complete_pct_type',
                                    'proj_short_name', 'acct_id', 'orig_proj_id', 'source_proj_id', 'base_type_id',
                                    'clndr_id',
                                    'sum_base_proj_id', 'task_code_base', 'task_code_step', 'priority_num',
                                    'wbs_max_sum_level',
                                    'strgy_priority_num', 'last_checksum', 'critical_drtn_hr_cnt', 'def_cost_per_qty',
                                    'last_recalc_date', 'plan_start_date', 'plan_end_date', 'scd_end_date', 'add_date',
                                    'last_tasksum_date', 'fcst_start_date', 'def_duration_type', 'task_code_prefix',
                                    'guid',
                                    'def_qty_type', 'add_by_name', 'web_local_root_path', 'proj_url', 'def_rate_type',
                                    'add_act_remain_flag', 'act_this_per_link_flag', 'def_task_type',
                                    'act_pct_link_flag',
                                    'critical_path_type', 'task_code_prefix_flag', 'def_rollup_dates_flag',
                                    'use_project_baseline_flag', 'rem_target_link_flag', 'reset_planned_flag',
                                    'allow_neg_act_flag', 'sum_assign_level', 'last_fin_dates_id',
                                    'last_baseline_update_date',
                                    'cr_external_key', 'apply_actuals_date', 'location_id', 'loaded_scope_level',
                                    'export_flag',
                                    'new_fin_dates_id', 'baselines_to_export', 'baseline_names_to_export',
                                    'next_data_date',
                                    'close_period_flag', 'sum_refresh_date', 'trsrcsum_loaded'])

    data = ['7607', '1', 'Y', 'Y', 'Y', 'N', 'Y', 'N', 'N', 'Y', '.', 'CP_Drtn', '',
            '', '', '', '', '6591', '', '1000', '10', '10', '2', '500', '', '0', '0,0000', '2021-10-01 00:00',
            '2018-04-17 00:00', '', '2021-09-03 15:00', '2022-05-13 12:44', '', '', 'DT_FixedDUR2', 'A',
            'WbTOzpjnlke//g4t+DA/Qg', 'QT_Hour', 'admin', '', '', 'COST_PER_QTY', 'N', 'Y', 'TT_Task', 'N',
            'CT_TotFloat', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'SL_Taskrsrc', '', '', '', '', '', '7', 'Y', '', '', '', '',
            '', '1899-12-30 00:00']

    proj_df = proj_df.append(pd.Series(data=dict(zip(proj_df.columns, data)), name=0), ignore_index=False)

    proj_df.loc[0, 'proj_short_name'] = project_name
    proj_df.loc[0, 'last_recalc_date'] = '2022-05-18 12:00'
    proj_df.loc[0, 'add_date'] = '2022-05-18 12:00'
    proj_df.loc[0, 'plan_start_date'] = proj_start_date
    proj_df.loc[0, 'plan_end_date'] = proj_finish_date
    proj_df.loc[0, 'scd_end_date'] = proj_finish_date

    # Create DataFrame with the structure of WBS in XER
    wbs_df = pd.DataFrame(
        columns=['wbs_id', 'proj_id', 'obs_id', 'seq_num', 'est_wt', 'proj_node_flag', 'sum_data_flag',
                 'status_code', 'wbs_short_name', 'wbs_name', 'phase_id', 'parent_wbs_id', 'ev_user_pct',
                 'ev_etc_user_value', 'orig_cost', 'indep_remain_total_cost', 'ann_dscnt_rate_pct',
                 'dscnt_period_type', 'indep_remain_work_qty', 'anticip_start_date', 'anticip_end_date',
                 'ev_compute_type', 'ev_etc_compute_type', 'guid', 'tmpl_guid', 'plan_open_state'])

    data = ['40210', '7607', '1992', '800', '1', 'Y', 'N', 'WS_Open', project_name,
            project_name + ' level 4', '', '3063', '6', '0,88', '0,0000', '0,0000', '', '', '', '', '',
            'EC_Cmp_pct', 'EE_Rem_hr', 'E8lkFHGaZ06YWRi+YrpV9A', '', '']
    wbs_df = wbs_df.append(pd.Series(data=dict(zip(wbs_df.columns, data)), name=0), ignore_index=False)

    proj_wbs_id = wbs_df.loc[0, 'wbs_id']

    clndr_df = pd.DataFrame(columns=['clndr_id', 'default_flag', 'clndr_name', 'proj_id', 'base_clndr_id',
                                     'last_chng_date', 'clndr_type', 'day_hr_cnt', 'week_hr_cnt', 'month_hr_cnt',
                                     'year_hr_cnt', 'rsrc_private', 'clndr_data'])

    data = ['6591', 'N', '24-hours', '', '', '2019-09-04 00:00', 'CA_Base', '24', '168', '720', '8760', 'N',
            '(0||CalendarData()(  (0||DaysOfWeek()(    (0||1()(      (0||0(s|00:00|f|00:00)())))    (0||2()(      '
            '(0||0(s|00:00|f|00:00)())))    (0||3()(      (0||0(s|00:00|f|00:00)())))    (0||4()(      '
            '(0||0(s|00:00|f|00:00)())))    (0||5()(      (0||0(s|00:00|f|00:00)())))    (0||6()(      '
            '(0||0(s|00:00|f|00:00)())))    (0||7()(      (0||0(s|00:00|f|00:00)())))))  (0||VIEW(ShowTotal|Y)())  '
            '(0||Exceptions()(    (0||0(d|41464)(      (0||0(s|00:00|f|00:00)())))    (0||1(d|43101)())))))']

    clndr_df = clndr_df.append(pd.Series(data=dict(zip(clndr_df.columns, data)), name=0), ignore_index=False)

    # Create DataFrame with the structure of resource columns in XER
    resources_df = pd.DataFrame(columns=['rsrc_id', 'parent_rsrc_id', 'clndr_id', 'role_id', 'shift_id', 'user_id',
                                         'pobs_id', 'guid', 'rsrc_seq_num', 'email_addr', 'employee_code',
                                         'office_phone',
                                         'other_phone', 'rsrc_name', 'rsrc_short_name', 'rsrc_title_name',
                                         'def_qty_per_hr',
                                         'cost_qty_type', 'ot_factor', 'active_flag', 'auto_compute_act_flag',
                                         'def_cost_qty_link_flag', 'ot_flag', 'curr_id', 'unit_id', 'rsrc_type',
                                         'location_id',
                                         'rsrc_notes', 'load_tasks_flag', 'level_flag', 'last_checksum'])

    # Add example of data row to the DataFrame
    data = ['6899', '', '6591', '', '', '', '', 'lfmGLf6l30qiQW01nlms4Q', '0', '', '', '', '',
            'Labor', '!Lab', '', '24', 'QT_Hour', '', 'Y', 'N', 'N', 'N', '18', '', 'RT_Labor',
            '', '', '', '', '']
    resources_df = resources_df.append(pd.Series(data=dict(zip(resources_df.columns, data)), name=0),
                                       ignore_index=False)

    # Get resources info
    resources_lst = []
    for x in electro_df['workers']:
        resources_lst.extend(list(x.keys()))
    resources_lst = list(set(resources_lst))

    resources_lst = [x.split('_')[0] for x in resources_lst]
    resources_info = list(
        zip([x[:3].upper() for x in resources_lst], resources_lst, range(6900, 6900 + len(resources_lst))))

    ex_res_data = resources_df.loc[0].to_dict()
    for res in resources_info:
        # print(res[0], res[1], res[2])
        res_data = ex_res_data.copy()
        res_data['rsrc_id'] = res[2]
        res_data['rsrc_name'] = res[1]
        res_data['rsrc_short_name'] = res[0]
        res_data['guid'] = str(res_data['guid']) + str(res[2])

        resources_df = resources_df.append(pd.Series(data=res_data, name=resources_df.index[-1] + 1), ignore_index=True)

    resources_df = resources_df.loc[:][1:]
    resources_df = resources_df.reset_index(drop=True)

    # Create DataFrame with the structure of task columns in XER
    tasks_df = pd.DataFrame(columns=XER_STRUCTURE)

    # Add example of data row to the DataFrame
    data = ['161234', '7607', '40227', '6591', '100', 'N', '1', 'N', 'N', 'CP_Drtn', 'TT_Task', 'DT_FixedDUR2',
            'TK_Complete', 'SPS10-1010', 'Ex_task', '', '', '', '0', '188600', '0', '188600',
            '336', '0', '0', '0', '', '2016-10-15 00:00', '2016-10-17 00:00', '2021-09-03 15:00', '2021-09-03 15:00',
            '', '2021-04-01 00:00', '2021-04-01 00:00', '', '', '2016-10-15 00:00', '2016-10-17 00:00', '', '', '',
            'PT_Normal', '', '', '', '', 'bQ0hxUuvikOkj0Ox6SnyAA', '', '', '', 'N', '188600', '0', '', '',
            '2022-05-13 12:44', '2022-05-13 12:44', 'Aleksandrov.YaO', 'Roslyakov.RI', '']

    tasks_df = tasks_df.append(pd.Series(data=dict(zip(tasks_df.columns, data)), name=0),
                               ignore_index=False)

    create_user = 'AutoScheduling'
    update_user = 'AutoScheduling'
    create_date = '2022-05-27 00:00'
    update_date = '2022-05-27 00:00'

    # Get tasks info
    tasks_info = []
    start_task_id = 161234
    for i in range(len(electro_df)):
        tasks_info.append({'id': start_task_id + i, 'task_code': electro_df.task_id[i], 'name': electro_df.task_name[i],
                           'start': electro_df.start[i] + ' 00:00', 'finish': electro_df.finish[i] + ' 23:59'})

    ex_res_data = tasks_df.loc[0].to_dict()
    for task in tasks_info:
        if 'start' in task['name'] or 'finish' in task['name']:
            continue
        task_data = ex_res_data.copy()
        task_data['task_id'] = task['id']
        task_data['task_code'] = task['task_code']
        task_data['task_name'] = task['name']
        task_data['guid'] = res_data['guid'] + str(task['id'])
        task_data['wbs_id'] = proj_wbs_id

        start_date = task['start'].split('.')[0][:-3]
        end_date = task['finish'].split('.')[0][:-3]

        task_data['act_start_date'] = start_date
        task_data['act_end_date'] = end_date
        task_data['late_start_date'] = start_date
        task_data['late_end_date'] = end_date
        task_data['early_start_date'] = start_date
        task_data['early_end_date'] = end_date
        task_data['target_start_date'] = start_date
        task_data['target_end_date'] = end_date

        task_data['create_date'] = create_date
        task_data['update_date'] = update_date
        task_data['create_user'] = create_user
        task_data['update_user'] = update_user

        tasks_df = tasks_df.append(pd.Series(data=task_data, name=tasks_df.index[-1] + 1), ignore_index=True)

    tasks_df = tasks_df.loc[:][1:]
    tasks_df = tasks_df.reset_index(drop=True)

    tasks_df = tasks_df.set_index('task_code')

    # Create DataFrame with the structure of tasks connectios columns in XER
    taskpred_df = pd.DataFrame(
        columns=['task_pred_id', 'task_id', 'pred_task_id', 'proj_id', 'pred_proj_id', 'pred_type',
                 'lag_hr_cnt', 'float_path', 'aref', 'arls'])

    # Add example of data row to the DataFrame
    data = ['142804', '161235', '161234', '7607', '7607', 'PR_SS', '24', '', '', '']

    taskpred_df = taskpred_df.append(pd.Series(data=dict(zip(taskpred_df.columns, data)), name=0),
                                     ignore_index=False)

    # electro_df['workers'] = [ast.literal_eval(x) for x in electro_df.workers]
    # electro_df['successors'] = [ast.literal_eval(x) for x in electro_df.successors]

    connections = []
    for i in electro_df.index:
        pred_task_code = electro_df.loc[i, 'task_id']
        pred_task_id = tasks_df.loc[pred_task_code, 'task_id']
        for succ_pair in electro_df.loc[i, 'successors']:
            if type(succ_pair) != list:
                succ_pair = ast.literal_eval(succ_pair)
            connections.append(
                prepare_tasks_connection(tasks_df.loc[int(succ_pair[0]), 'task_id'], pred_task_id, succ_pair[1]))

    connection_id = 170180
    ex_conn_data = taskpred_df.loc[0].to_dict()

    for connection in connections:
        conn_data = ex_conn_data.copy()
        conn_data['task_pred_id'] = connection_id
        conn_data['task_id'] = connection['task_id']
        conn_data['pred_task_id'] = connection['pred_task_id']
        conn_data['pred_type'] = connection['pred_type']
        conn_data['lag_hr_cnt'] = '0'

        connection_id += 1

        taskpred_df = taskpred_df.append(pd.Series(data=conn_data, name=taskpred_df.index[-1] + 1),
                                         ignore_index=True)

    taskpred_df = taskpred_df.loc[:][1:]
    taskpred_df = taskpred_df.reset_index(drop=True)

    resources_df = resources_df.set_index('rsrc_name')

    # Create DataFrame with the structure of resource assignment columns in XER
    tasks_resources_df = pd.DataFrame(columns=['taskrsrc_id', 'task_id', 'proj_id', 'cost_qty_link_flag',
                                               'role_id', 'acct_id', 'rsrc_id', 'pobs_id', 'skill_level', 'remain_qty',
                                               'target_qty', 'remain_qty_per_hr', 'target_lag_drtn_hr_cnt',
                                               'target_qty_per_hr', 'act_ot_qty', 'act_reg_qty', 'relag_drtn_hr_cnt',
                                               'ot_factor', 'cost_per_qty', 'target_cost', 'act_reg_cost',
                                               'act_ot_cost',
                                               'remain_cost', 'act_start_date', 'act_end_date', 'restart_date',
                                               'reend_date',
                                               'target_start_date', 'target_end_date', 'rem_late_start_date',
                                               'rem_late_end_date', 'rollup_dates_flag', 'target_crv', 'remain_crv',
                                               'actual_crv', 'ts_pend_act_end_flag', 'guid', 'rate_type',
                                               'act_this_per_cost',
                                               'act_this_per_qty', 'curv_id', 'rsrc_type', 'cost_per_qty_source_type',
                                               'create_user', 'create_date', 'has_rsrchours', 'taskrsrc_sum_id'])

    # Add example of data row to the DataFrame
    data = ['179591', '161234', '7607', 'Y', '', '', '6901', '', '', '0', '4', '0', '0', '0', '0', '4', '0',
            '', '', '4', '4', '0', '0', '2016-10-15 01:12', '2016-10-17 00:01', '', '',
            '2016-10-15 01:12', '2016-10-17 00:01', '', '', 'Y', '', '', '', 'N', 'EXOGiioaRE287vsNT70HOw',
            'COST_PER_QTY',
            '4', '4', '', 'RT_Labor', 'ST_Rsrc', 'Aleksandrov.YaO', '2022-05-13 12:44']

    tasks_resources_df = tasks_resources_df.append(
        pd.Series(data=dict(zip(tasks_resources_df.columns.str.encode('utf-8'), data)), name=0),
        ignore_index=False)

    ex_res_data = tasks_resources_df.loc[0].to_dict()
    taskrsrc_id = 179590
    series_name = -1

    for i in range(1, len(electro_df) - 1):
        task_name = electro_df.task_id[i]
        task_id = tasks_df.task_id[task_name]
        task_start = tasks_df.act_start_date[task_name]
        task_end = tasks_df.act_end_date[task_name]
        for res in electro_df.workers[i]:
            res_id = resources_df.rsrc_id[res.split('_')[0]]
            res_vol = electro_df.workers[i][res]
            taskrsrc_id = taskrsrc_id + 1
            series_name = series_name + 1
            task_res_data = get_task_rsrc_str(ex_res_data, series_name, taskrsrc_id, task_id, task_start,
                                              task_end, res_id, res_vol, create_date, update_date,
                                              create_user, update_user)
            tasks_resources_df = tasks_resources_df.append(task_res_data, ignore_index=True)

    tasks_resources_df = tasks_resources_df.loc[:][1:]
    tasks_resources_df = tasks_resources_df.reset_index(drop=True)

    resources_df = resources_df.reset_index()
    tasks_df = tasks_df.reset_index()

    tasks_df = tasks_df.loc[:, XER_STRUCTURE]

    resources_df = resources_df.loc[:, ['rsrc_id', 'parent_rsrc_id', 'clndr_id', 'role_id', 'shift_id', 'user_id',
                                        'pobs_id', 'guid', 'rsrc_seq_num', 'email_addr', 'employee_code',
                                        'office_phone', 'other_phone', 'rsrc_name', 'rsrc_short_name',
                                        'rsrc_title_name', 'def_qty_per_hr', 'cost_qty_type', 'ot_factor',
                                        'active_flag', 'auto_compute_act_flag', 'def_cost_qty_link_flag', 'ot_flag',
                                        'curr_id', 'unit_id', 'rsrc_type', 'location_id', 'rsrc_notes',
                                        'load_tasks_flag', 'level_flag', 'last_checksum']]

    df_lst = [proj_df, clndr_df, wbs_df, resources_df, tasks_df, taskpred_df, tasks_resources_df]
    lbls_lst = ['PROJECT', 'CALENDAR', 'PROJWBS', 'RSRC', 'TASK', 'TASKPRED', 'TASKRSRC']

    xer_base_file = open(base_xer_file, mode='r', encoding='cp1251')
    file_to_save = open(path_to_save, mode='w')
    file_to_save.writelines(xer_base_file.readlines())
    xer_base_file.close()

    for i in range(len(df_lst)):
        file_to_save.write('%T' + '\t' + lbls_lst[i] + '\n')
        file_to_save.write('%F' + '\t' + '\t'.join([str(x) for x in list(df_lst[i].columns)]) + '\n')
        for row in df_lst[i].astype(str).values:
            file_to_save.write('%R' + '\t' + '\t'.join(row) + '\n')

    file_to_save.write('%E')
    file_to_save.close()
