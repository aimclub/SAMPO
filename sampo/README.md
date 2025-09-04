# Docstring Reference / Справочник docstring'ов

Automatically collected docstrings from the SAMPO package. / Автоматически собранные docstring'и из пакета SAMPO.

## Table of Contents / Оглавление

- [api](#api)
  - [genetic_api.py](#api-genetic_apipy)
- [backend](#backend)
  - [__init__.py](#backend-__init__py)
  - [default.py](#backend-defaultpy)
  - [multiproc.py](#backend-multiprocpy)
- [generator](#generator)
  - [base.py](#generator-basepy)
- [generator/config](#generatorconfig)
  - [worker_req.py](#generatorconfig-worker_reqpy)
- [generator/environment](#generatorenvironment)
  - [contractor.py](#generatorenvironment-contractorpy)
  - [contractor_by_wg.py](#generatorenvironment-contractor_by_wgpy)
  - [landscape.py](#generatorenvironment-landscapepy)
- [generator/pipeline](#generatorpipeline)
  - [__init__.py](#generatorpipeline-__init__py)
  - [cluster.py](#generatorpipeline-clusterpy)
  - [extension.py](#generatorpipeline-extensionpy)
  - [project.py](#generatorpipeline-projectpy)
- [hybrid](#hybrid)
  - [cycle.py](#hybrid-cyclepy)
  - [population.py](#hybrid-populationpy)
- [landscape_config](#landscape_config)
  - [material_request.py](#landscape_config-material_requestpy)
  - [road_workload.py](#landscape_config-road_workloadpy)
- [native](#native)
  - [setup.py](#native-setuppy)
- [pipeline](#pipeline)
  - [__init__.py](#pipeline-__init__py)
  - [base.py](#pipeline-basepy)
  - [default.py](#pipeline-defaultpy)
  - [delegating.py](#pipeline-delegatingpy)
  - [lag_optimization.py](#pipeline-lag_optimizationpy)
  - [preparation.py](#pipeline-preparationpy)
- [root](#root)
  - [base.py](#root-basepy)
  - [docstring_readme.py](#root-docstring_readmepy)
- [scheduler](#scheduler)
  - [base.py](#scheduler-basepy)
  - [generic.py](#scheduler-genericpy)
  - [native_wrapper.py](#scheduler-native_wrapperpy)
- [scheduler/genetic](#schedulergenetic)
  - [base.py](#schedulergenetic-basepy)
  - [converter.py](#schedulergenetic-converterpy)
  - [operators.py](#schedulergenetic-operatorspy)
  - [schedule_builder.py](#schedulergenetic-schedule_builderpy)
  - [utils.py](#schedulergenetic-utilspy)
- [scheduler/heft](#schedulerheft)
  - [base.py](#schedulerheft-basepy)
  - [prioritization.py](#schedulerheft-prioritizationpy)
- [scheduler/lft](#schedulerlft)
  - [base.py](#schedulerlft-basepy)
  - [prioritization.py](#schedulerlft-prioritizationpy)
  - [time_computaion.py](#schedulerlft-time_computaionpy)
- [scheduler/multi_agency](#schedulermulti_agency)
  - [block_generator.py](#schedulermulti_agency-block_generatorpy)
  - [block_graph.py](#schedulermulti_agency-block_graphpy)
  - [block_validation.py](#schedulermulti_agency-block_validationpy)
  - [exception.py](#schedulermulti_agency-exceptionpy)
  - [multi_agency.py](#schedulermulti_agency-multi_agencypy)
- [scheduler/resource](#schedulerresource)
  - [average_req.py](#schedulerresource-average_reqpy)
  - [base.py](#schedulerresource-basepy)
  - [coordinate_descent.py](#schedulerresource-coordinate_descentpy)
  - [full_scan.py](#schedulerresource-full_scanpy)
  - [identity.py](#schedulerresource-identitypy)
- [scheduler/resources_in_time](#schedulerresources_in_time)
  - [average_binary_search.py](#schedulerresources_in_time-average_binary_searchpy)
- [scheduler/selection](#schedulerselection)
  - [metrics.py](#schedulerselection-metricspy)
  - [neural_net.py](#schedulerselection-neural_netpy)
  - [validation.py](#schedulerselection-validationpy)
- [scheduler/timeline](#schedulertimeline)
  - [base.py](#schedulertimeline-basepy)
  - [general_timeline.py](#schedulertimeline-general_timelinepy)
  - [hybrid_supply_timeline.py](#schedulertimeline-hybrid_supply_timelinepy)
  - [just_in_time_timeline.py](#schedulertimeline-just_in_time_timelinepy)
  - [momentum_timeline.py](#schedulertimeline-momentum_timelinepy)
  - [platform_timeline.py](#schedulertimeline-platform_timelinepy)
  - [to_start_supply_timeline.py](#schedulertimeline-to_start_supply_timelinepy)
  - [utils.py](#schedulertimeline-utilspy)
  - [zone_timeline.py](#schedulertimeline-zone_timelinepy)
- [scheduler/topological](#schedulertopological)
  - [base.py](#schedulertopological-basepy)
- [scheduler/utils](#schedulerutils)
  - [__init__.py](#schedulerutils-__init__py)
  - [local_optimization.py](#schedulerutils-local_optimizationpy)
  - [multi_contractor.py](#schedulerutils-multi_contractorpy)
  - [obstruction.py](#schedulerutils-obstructionpy)
  - [time_computaion.py](#schedulerutils-time_computaionpy)
- [schemas](#schemas)
  - [apply_queue.py](#schemas-apply_queuepy)
  - [contractor.py](#schemas-contractorpy)
  - [exceptions.py](#schemas-exceptionspy)
  - [graph.py](#schemas-graphpy)
  - [identifiable.py](#schemas-identifiablepy)
  - [interval.py](#schemas-intervalpy)
  - [landscape.py](#schemas-landscapepy)
  - [landscape_graph.py](#schemas-landscape_graphpy)
  - [project.py](#schemas-projectpy)
  - [requirements.py](#schemas-requirementspy)
  - [resources.py](#schemas-resourcespy)
  - [schedule.py](#schemas-schedulepy)
  - [schedule_spec.py](#schemas-schedule_specpy)
  - [scheduled_work.py](#schemas-scheduled_workpy)
  - [serializable.py](#schemas-serializablepy)
  - [sorted_list.py](#schemas-sorted_listpy)
  - [structure_estimator.py](#schemas-structure_estimatorpy)
  - [time.py](#schemas-timepy)
  - [time_estimator.py](#schemas-time_estimatorpy)
  - [types.py](#schemas-typespy)
  - [utils.py](#schemas-utilspy)
  - [works.py](#schemas-workspy)
  - [zones.py](#schemas-zonespy)
- [structurator](#structurator)
  - [base.py](#structurator-basepy)
  - [delete_graph_node.py](#structurator-delete_graph_nodepy)
  - [graph_insertion.py](#structurator-graph_insertionpy)
  - [insert_wu.py](#structurator-insert_wupy)
  - [light_modifications.py](#structurator-light_modificationspy)
  - [prepare_wg_copy.py](#structurator-prepare_wg_copypy)
- [userinput/parser](#userinputparser)
  - [contractor_type.py](#userinputparser-contractor_typepy)
  - [csv_parser.py](#userinputparser-csv_parserpy)
  - [exception.py](#userinputparser-exceptionpy)
  - [general_build.py](#userinputparser-general_buildpy)
  - [history.py](#userinputparser-historypy)
- [utilities](#utilities)
  - [base_opt.py](#utilities-base_optpy)
  - [collections_util.py](#utilities-collections_utilpy)
  - [datetime_util.py](#utilities-datetime_utilpy)
  - [linked_list.py](#utilities-linked_listpy)
  - [name_mapper.py](#utilities-name_mapperpy)
  - [priority.py](#utilities-prioritypy)
  - [priority_queue.py](#utilities-priority_queuepy)
  - [resource_usage.py](#utilities-resource_usagepy)
  - [schedule.py](#utilities-schedulepy)
  - [serializers.py](#utilities-serializerspy)
  - [validation.py](#utilities-validationpy)
- [utilities/sampler](#utilitiessampler)
  - [__init__.py](#utilitiessampler-__init__py)
  - [requirements.py](#utilitiessampler-requirementspy)
  - [types.py](#utilitiessampler-typespy)
  - [works.py](#utilitiessampler-workspy)
- [utilities/visualization](#utilitiesvisualization)
  - [__init__.py](#utilitiesvisualization-__init__py)
  - [base.py](#utilitiesvisualization-basepy)
  - [resources.py](#utilitiesvisualization-resourcespy)
  - [schedule.py](#utilitiesvisualization-schedulepy)
  - [work_graph.py](#utilitiesvisualization-work_graphpy)

## <a id="api"></a>api

### <a id="api-genetic_apipy"></a>[genetic_api.py](api/genetic_api.py)

#### Classes / Классы
- **FitnessFunction**

    ```
    Base class for description of different fitness functions.
    ```

- **Individual**

- **ScheduleGenerationScheme**


## <a id="backend"></a>backend

### <a id="backend-__init__py"></a>[__init__.py](backend/__init__.py)

#### Classes / Классы
- **ComputationalBackend**


### <a id="backend-defaultpy"></a>[default.py](backend/default.py)

#### Classes / Классы
- **DefaultComputationalBackend**


### <a id="backend-multiprocpy"></a>[multiproc.py](backend/multiproc.py)

#### Classes / Классы
- **MultiprocessingComputationalBackend**

    ```
    Backend that computes chromosomes in parallel.
    
    Бэкенд, вычисляющий хромосомы параллельно.
    ```

#### Functions / Функции
- **scheduler_info_initializer**

    ```
    Initialize global data for child processes.
    
    Инициализирует глобальные данные для дочерних процессов.
    
    Args:
        wg (WorkGraph): Work graph to schedule.
            Граф работ для планирования.
        contractors (list[Contractor]): Available contractors.
            Доступные подрядчики.
        landscape (LandscapeConfiguration): Landscape configuration.
            Конфигурация ландшафта.
        spec (ScheduleSpec): Scheduling specification.
            Спецификация планирования.
        selection_size (int): Size of population selection.
            Размер выборки популяции.
        mutate_order (float): Probability of order mutation.
            Вероятность мутации порядка.
        mutate_resources (float): Probability of resource mutation.
            Вероятность мутации ресурсов.
        mutate_zones (float): Probability of zone mutation.
            Вероятность мутации зон.
        deadline (Time | None): Scheduling deadline.
            Дедлайн планирования.
        weights (list[int] | None): Weights for chromosome types.
            Веса типов хромосом.
        init_chromosomes (dict[str, tuple[ChromosomeType, float, ScheduleSpec]]):
            Cached initial chromosomes.
            Кэш начальных хромосом.
        assigned_parent_time (Time): Parent schedule time.
            Время родительского расписания.
        fitness_weights (tuple[int | float, ...]): Fitness weights.
            Веса функции приспособленности.
        rand (Random | None): Random generator.
            Генератор случайных чисел.
        work_estimator_recreate_params (tuple | None): Params to recreate
            estimator.
            Параметры для пересоздания оценщика.
        sgs_type (ScheduleGenerationScheme): Schedule generation scheme.
            Схема генерации расписания.
        only_lft_initialization (bool): Use only LFT initialization.
            Использовать только LFT инициализацию.
        is_multiobjective (bool): Multiobjective optimization flag.
            Флаг многокритериальной оптимизации.
    ```


## <a id="generator"></a>generator

### <a id="generator-basepy"></a>[base.py](generator/base.py)

#### Classes / Классы
- **SimpleSynthetic**

    ```
    Simplified interface for synthetic data generation.
    
    Упрощенный интерфейс для генерации синтетических данных.
    ```


## <a id="generatorconfig"></a>generator/config

### <a id="generatorconfig-worker_reqpy"></a>[worker_req.py](generator/config/worker_req.py)

#### Functions / Функции
- **get_borehole_volume**

    ```
    Compute volume multiplier based on boreholes.
    
    Вычисляет множитель объёма в зависимости от буровых.
    
    Args:
        borehole_count (int): Number of boreholes.
            Количество буровых.
        base (tuple[float, float]): Base volumes independent and dependent on
            boreholes.
            Базовые объёмы, независимые и зависящие от буровых.
    
    Returns:
        float: Volume multiplier.
            Множитель объёма.
    ```

- **mul_borehole_volume**

    ```
    Scale requirements by borehole count.
    
    Масштабирует требования по числу буровых.
    
    Args:
        req_list (list[WorkerReq]): Requirements to scale.
            Требования для масштабирования.
        borehole_count (int): Number of boreholes.
            Количество буровых.
        base (tuple[float, float]): Base volumes independent and dependent on
            boreholes.
            Базовые объёмы, независимые и зависящие от буровых.
    
    Returns:
        list[WorkerReq]: Scaled requirements.
            Масштабированные требования.
    ```

- **mul_volume_reqs**

    ```
    Scale only volume of requirements.
    
    Масштабирует только объём требований.
    
    Args:
        req_list (list[WorkerReq]): Requirements to scale.
            Требования для масштабирования.
        scalar (float): Scaling factor.
            Коэффициент масштабирования.
        new_name (str | None): Optional new name.
            Необязательное новое имя.
    
    Returns:
        list[WorkerReq]: Scaled requirements.
            Масштабированные требования.
    ```

- **scale_reqs**

    ```
    Scale requirements by scalar.
    
    Масштабирует требования по коэффициенту.
    
    Args:
        req_list (list[WorkerReq]): Requirements to scale.
            Требования для масштабирования.
        scalar (float): Scaling factor.
            Коэффициент масштабирования.
        new_name (str | None): Optional new name.
            Необязательное новое имя.
    
    Returns:
        list[WorkerReq]: Scaled requirements.
            Масштабированные требования.
    ```


## <a id="generatorenvironment"></a>generator/environment

### <a id="generatorenvironment-contractorpy"></a>[contractor.py](generator/environment/contractor.py)

#### Functions / Функции
- **_dict_subtract**

    ```
    :param d: dict[str:
    :param float]:
    :param subtractor: float:
    ```

- **_get_stochastic_counts**

    ```
    Return random quantity of each type of resources. Random value is gotten from Gaussian distribution
    
    :param pack_count: The number of resource sets
    :param sigma_scaler: parameter to calculate the scatter by Gaussian distribution with mean=0 amount from the
    transferred proportions
    :param proportions: proportions of quantity for contractor resources to be scaled by pack_worker_count
    :param available_types: Worker types for generation,
    if a subset of worker_proportions is used, if None, all worker_proportions are used
    :param rand: Number generator with a fixed seed, or None for no fixed seed
    ```

- **get_contractor**

    ```
    Generates a contractor for a synthetic graph for a given resource scalar and generation parameters
    
    :param pack_worker_count: The number of resource sets
    :param sigma_scaler: parameter to calculate the scatter by Gaussian distribution with mean=0 amount from the
    transferred proportions
    :param worker_proportions: proportions of quantity for contractor resources to be scaled by pack_worker_count
    :param available_worker_types: Worker types for generation,
    if a subset of worker_proportions is used, if None, all worker_proportions are used
    :param rand: Number generator with a fixed seed, or None for no fixed seed
    :param contractor_id: generated contractor's id
    :param contractor_name: generated contractor's name
    :returns: the contractor
    ```

- **get_contractor_with_equal_proportions**

    ```
    Generates a contractors list of specified length with specified capacities
    
    :param number_of_workers_in_contractors: How many workers of all each contractor contains in itself.
    One int for all or list[int] for each contractor. If list, its length should be equal to number_of_contractors
    :param number_of_contractors: Number of generated contractors.
    :returns: list with contractors
    ```


### <a id="generatorenvironment-contractor_by_wgpy"></a>[contractor_by_wg.py](generator/environment/contractor_by_wg.py)

#### Classes / Классы
- **ContractorGenerationMethod**

#### Functions / Функции
- **_value_by_req**

    ```
    Sets the function by which the number for the function of searching for a contractor by the graph of works
    is determined by the given parameter
    
    :param method: type the specified parameter: min ~ min_count, max ~ max_count, avg ~ (min_count + max_count) / 2
    :param req: the Worker Req
    :return:
    ```

- **get_contractor_by_wg**

    ```
    Creates a pool of contractor resources based on job requirements, selecting the maximum specified parameter
    
    :param wg: The graph of works for which it is necessary to find a set of resources
    :param scaler: Multiplier for the number of resources in the contractor
    :param method: type the specified parameter: min ~ min_count, max ~ max_count, avg ~ (min_count + max_count) / 2
    :param contractor_id: generated contractor's id
    :param contractor_name: generated contractor's name
    :return: the contractor capable of completing given `WorkGraph`
    ```


### <a id="generatorenvironment-landscapepy"></a>[landscape.py](generator/environment/landscape.py)

#### Functions / Функции
- **get_landscape_by_wg**

    ```
    Generate landscape based on a work graph.
    
    Генерирует ландшафт на основе графа работ.
    
    Args:
        wg (WorkGraph): Work graph.
            Граф работ.
        rnd (random.Random): Random generator.
            Генератор случайных чисел.
    
    Returns:
        LandscapeConfiguration: Generated landscape.
            Сгенерированный ландшафт.
    ```

- **setup_landscape**

    ```
    Build landscape configuration from provided data.
    
    Создает конфигурацию ландшафта из предоставленных данных.
    
    Args:
        platforms_info (dict[str, dict[str, int]]): Material counts on
            platforms.
            Количество материалов на площадках.
        warehouses_info (dict[str, list[dict[str, int], list[tuple[str, dict[str, int]]]]]):
            Warehouses with materials and vehicles.
            Склады с материалами и техникой.
        roads_info (dict[str, list[tuple[str, float, int]]]): Road connections
            between platforms.
            Дорожные соединения между площадками.
    
    Returns:
        LandscapeConfiguration: Configured landscape.
            Настроенная конфигурация ландшафта.
    ```


## <a id="generatorpipeline"></a>generator/pipeline

### <a id="generatorpipeline-__init__py"></a>[__init__.py](generator/pipeline/__init__.py)

#### Classes / Классы
- **SyntheticGraphType**

    ```
    Describe available types of synthetic graph
    
    * PARALLEL - work graph dominated by parallel works
    * SEQUENTIAL - work graph dominated by sequential works
    * GENERAL - work graph, including sequential and parallel works, it is similar to the real work graphs
    ```


### <a id="generatorpipeline-clusterpy"></a>[cluster.py](generator/pipeline/cluster.py)

#### Functions / Функции
- **_add_addition_work**

    ```
    Return answer, if addition work will be added
    ```

- **_get_boreholes**

- **_get_boreholes_equipment_general**

- **_get_boreholes_equipment_group**

- **_get_boreholes_equipment_shared**

- **_get_engineering_preparation**

- **_get_handing_stage**

- **_get_pipe_lines**

- **_get_power_lines**

- **_get_roads**

- **get_cluster_works**

    ```
    Create works for developing a field cluster.
    
    Создаёт работы для разработки кластера месторождения.
    
    Args:
        cluster_name (str): Name of the cluster.
            Имя кластера.
        pipe_nodes_count (int): Number of pipeline segments to other fields.
            Количество участков трубопровода к другим месторождениям.
        pipe_net_count (int): Number of pipes connecting boreholes.
            Число труб, соединяющих скважины.
        light_masts_count (int): Number of floodlight masts.
            Количество осветительных мачт.
        borehole_counts (list[int]): Number of boreholes in each group.
            Количество скважин в каждой группе.
        roads (dict[str, GraphNode] | None): Roads connecting to the central
            node, if any.
            Дороги, соединяющие с центральным узлом, если имеются.
        rand (Random | None): Random number generator.
            Генератор случайных чисел.
    
    Returns:
        tuple[GraphNode, dict[str, GraphNode], int]:
        Root node of the cluster, generated roads, and number of created
            nodes.
        Корневой узел кластера, созданные дороги и количество
            сформированных узлов.
    ```


### <a id="generatorpipeline-extensionpy"></a>[extension.py](generator/pipeline/extension.py)

#### Functions / Функции
- **_extend_str_fields**

- **_get_uniq_resource_kinds**

- **_get_uniq_work_names**

- **_update_resource_names**

- **_update_work_name**

- **extend_names**

    ```
    Increases the number of unique work names in WorkGraph
    
    :param uniq_activities:  the amount to which you need to increase
    :param wg: original WorkGraph
    :param rand: Number generator with a fixed seed, or None for no fixed seed
    :return: modified WorkGraph
    ```

- **extend_resources**

    ```
    Increases the number of unique resources in WorkGraph
    
    :param uniq_resources: the amount to which you need to increase
    :param wg: original WorkGraph
    :param rand: Number generator with a fixed seed, or None for no fixed seed
    :return: modified WorkGraph
    ```


### <a id="generatorpipeline-projectpy"></a>[project.py](generator/pipeline/project.py)

#### Functions / Функции
- **_general_graph_mode**

- **_get_cluster_graph**

- **_graph_mode_to_callable**

- **_parallel_graph_mode_get_root**

- **_sequence_graph_mode_get_root**

- **get_graph**

    ```
    Generate a synthetic work graph of the specified type.
    
    Генерирует синтетический граф работ указанного типа.
    
    Args:
        mode (SyntheticGraphType | None): Type of graph to generate.
            Тип генерируемого графа.
        cluster_name_prefix (str | None): Prefix used for cluster names.
            Префикс, используемый для имен кластеров.
        cluster_counts (int | None): Desired number of clusters in the graph.
            Требуемое число кластеров в графе.
        branching_probability (float | None): Probability of connecting a
            cluster to a non-sequential predecessor.
            Вероятность соединения кластера с неочередным предшественником.
        addition_cluster_probability (float | None): Probability of adding a
            slave cluster.
            Вероятность добавления подчинённого кластера.
        bottom_border (int | None): Minimum number of works in the graph.
            Минимальное количество работ в графе.
        top_border (int | None): Maximum number of works in the graph.
            Максимальное количество работ в графе.
        rand (Random | None): Random number generator.
            Генератор случайных чисел.
    
    Returns:
        WorkGraph: Generated work graph.
        WorkGraph: Сгенерированный граф работ.
    ```

- **get_small_graph**

    ```
    Create a small work graph with 30-50 vertices.
    
    Создаёт небольшой граф работ, содержащий 30–50 вершин.
    
    Args:
        cluster_name (str | None): Name of the initial cluster.
            Имя первого кластера.
        rand (Random | None): Random number generator.
            Генератор случайных чисел.
    
    Returns:
        WorkGraph: Work graph containing between 30 and 50 vertices.
        WorkGraph: Граф работ, включающий от 30 до 50 вершин.
    ```


## <a id="hybrid"></a>hybrid

### <a id="hybrid-cyclepy"></a>[cycle.py](hybrid/cycle.py)

#### Classes / Классы
- **CycleHybridScheduler**


### <a id="hybrid-populationpy"></a>[population.py](hybrid/population.py)

#### Classes / Классы
- **GeneticPopulationScheduler**

- **HeuristicPopulationScheduler**

- **PopulationScheduler**


## <a id="landscape_config"></a>landscape_config

### <a id="landscape_config-material_requestpy"></a>[material_request.py](landscape_config/material_request.py)

#### Functions / Функции
- **max_fill**

- **necessary_fill**


### <a id="landscape_config-road_workloadpy"></a>[road_workload.py](landscape_config/road_workload.py)

#### Functions / Функции
- **intensity**

- **static_workload**

    ```
    Calculate rate of theoretical road workload
    :param bandwidth:
    :param vehicle_num:
    :param length:
    :param max_velocity:
    :return:
    ```


## <a id="native"></a>native

### <a id="native-setuppy"></a>[setup.py](native/setup.py)

#### Classes / Классы
- **BuildFailed**

    ```
    Raised when C++ extension build fails.
    
    Возникает при сбое сборки C++-расширения.
    ```

- **CMakeExtension**

    ```
    Minimal CMake-based extension.
    
    Минимальное расширение на основе CMake.
    ```

- **ExtBuilder**

    ```
    Wrapper that converts build errors to BuildFailed.
    
    Обёртка, преобразующая ошибки сборки в BuildFailed.
    ```

- **build_ext**

    ```
    Custom build_ext command using CMake.
    
    Пользовательская команда build_ext с использованием CMake.
    ```

#### Functions / Функции
- **build**

    ```
    Prepare keyword arguments for building extensions.
    
    Обновляет параметры для сборки расширений.
    
    Args:
        setup_kwargs (dict): Keyword arguments passed to ``setup``.
            Аргументы, передаваемые в ``setup``.
    ```


## <a id="pipeline"></a>pipeline

### <a id="pipeline-__init__py"></a>[__init__.py](pipeline/__init__.py)

#### Classes / Классы
- **PipelineError**

    ```
    Raised when any pipeline error occurred.
    
    This is a kind of 'IllegalStateException', e.g. raising this
    indicates that the corresponding pipeline come to incorrect internal state.
    ```

- **PipelineType**

- **SchedulingPipeline**


### <a id="pipeline-basepy"></a>[base.py](pipeline/base.py)

#### Classes / Классы
- **InputPipeline**

    ```
    Base class to build different pipeline, that help to use the framework
    ```

- **SchedulePipeline**

    ```
    The part of pipeline, that manipulates with the whole entire schedule.
    ```


### <a id="pipeline-defaultpy"></a>[default.py](pipeline/default.py)

#### Classes / Классы
- **DefaultInputPipeline**

    ```
    Default pipeline simplifying framework usage.
    
    Стандартный конвейер, упрощающий использование фреймворка.
    ```

- **DefaultSchedulePipeline**

    ```
    Pipeline for processing generated schedules.
    
    Конвейер для обработки полученных расписаний.
    ```

#### Functions / Функции
- **contractors_can_perform_work_graph**

    ```
    Check if each work node has an eligible contractor.
    
    Проверяет, имеет ли каждый узел работ подходящего подрядчика.
    
    Args:
        contractors (list[Contractor]): Available contractors.
            Доступные подрядчики.
        wg (WorkGraph): Work graph to evaluate.
            Граф работ для оценки.
    
    Returns:
        bool: True if each node can be performed by at least one contractor.
        bool: True, если каждый узел может быть выполнен хотя бы одним
            подрядчиком.
    ```


### <a id="pipeline-delegatingpy"></a>[delegating.py](pipeline/delegating.py)

#### Classes / Классы
- **DelegatingScheduler**

    ```
    The layer between Generic Scheduler and end Scheduler.
    It's needed to change parametrization functions received from end Scheduler in a simple way.
    ```


### <a id="pipeline-lag_optimizationpy"></a>[lag_optimization.py](pipeline/lag_optimization.py)

#### Classes / Классы
- **LagOptimizationStrategy**


### <a id="pipeline-preparationpy"></a>[preparation.py](pipeline/preparation.py)

#### Classes / Классы
- **PreparationPipeline**


## <a id="root"></a>root

### <a id="root-basepy"></a>[base.py](base.py)

#### Classes / Классы
- **SAMPO**


### <a id="root-docstring_readmepy"></a>[docstring_readme.py](docstring_readme.py)

#### Functions / Функции
- **build_anchor**

    ```
    Create an anchor id from path parts. / Создаёт идентификатор якоря из частей пути.
    ```

- **collect_docstrings**

    ```
    Collect docstrings from package modules. / Собирает docstring'и из модулей пакета.
    ```

- **format_section**

    ```
    Format docstrings section preserving original text formatting.
    
    Форматирует секцию docstring'ов, сохраняя исходное форматирование текста.
    ```

- **generate_readme**

    ```
    Create README.md summarizing docstrings. / Создаёт README.md с перечнем docstring'ов.
    ```


## <a id="scheduler"></a>scheduler

### <a id="scheduler-basepy"></a>[base.py](scheduler/base.py)

#### Classes / Классы
- **Scheduler**

    ```
    Base class that implements scheduling logic.
    Базовый класс, реализующий логику планирования.
    ```

- **SchedulerType**

    ```
    Enumeration of available scheduler implementations.
    Перечисление доступных реализаций планировщика.
    ```


### <a id="scheduler-genericpy"></a>[generic.py](scheduler/generic.py)

#### Classes / Классы
- **GenericScheduler**

    ```
    Universal scheduler with customizable strategies.
    Универсальный планировщик с настраиваемыми стратегиями.
    ```

#### Functions / Функции
- **get_finish_time_default**

    ```
    Estimate finish time using default method.
    Оценивает время завершения стандартным методом.
    
    Args:
        node: Current graph node.
            Текущая вершина графа.
        worker_team: Worker team assigned to the node.
            Команда работников, назначенная на вершину.
        node2swork: Mapping of nodes to scheduled works.
            Отображение вершин в запланированные работы.
        spec: Work specification.
            Спецификация работы.
        assigned_parent_time: Parent start time.
            Время начала родителя.
        timeline: Timeline instance.
            Экземпляр временной шкалы.
        work_estimator: Work time estimator.
            Оценщик времени выполнения работ.
    
    Returns:
        Time: Estimated finish time.
            Оценка времени завершения.
    ```


### <a id="scheduler-native_wrapperpy"></a>[native_wrapper.py](scheduler/native_wrapper.py)

#### Classes / Классы
- **NativeWrapper**

    ```
    Interface to native genetic scheduling engine.
    Интерфейс к нативному движку генетического планирования.
    ```


## <a id="schedulergenetic"></a>scheduler/genetic

### <a id="schedulergenetic-basepy"></a>[base.py](scheduler/genetic/base.py)

#### Classes / Классы
- **GeneticScheduler**

    ```
    Hybrid scheduler combining heuristics and genetic search.
    
    Гибридный планировщик, сочетающий эвристику и генетический поиск.
    ```


### <a id="schedulergenetic-converterpy"></a>[converter.py](scheduler/genetic/converter.py)

#### Functions / Функции
- **convert_chromosome_to_schedule**

    ```
    Build schedule from chromosome.
    
    Формирует расписание из хромосомы.
    
    Args:
        chromosome (ChromosomeType): Source chromosome.
            Исходная хромосома.
        worker_pool (WorkerContractorPool): Available workers per contractor.
            Доступные работники у подрядчиков.
        index2node (dict[int, GraphNode]): Mapping of indices to nodes.
            Отображение индексов в узлы.
        index2contractor (dict[int, Contractor]): Mapping of indices to contractors.
            Отображение индексов в подрядчиков.
        index2zone (dict[int, str]): Mapping of indices to zones.
            Отображение индексов в зоны.
        worker_pool_indices (dict[int, dict[int, Worker]]): Worker lookup structure.
            Структура поиска рабочих.
        worker_name2index (dict[str, int]): Mapping of worker names to indices.
            Отображение имен рабочих в индексы.
        contractor2index (dict[str, int]): Mapping of contractor IDs to indices.
            Отображение идентификаторов подрядчиков в индексы.
        landscape (LandscapeConfiguration): Landscape configuration.
            Конфигурация ландшафта.
        timeline (Timeline | None): Existing timeline.
            Существующая временная шкала.
        assigned_parent_time (Time): Start time of parent schedule.
            Время начала родительского расписания.
        work_estimator (WorkTimeEstimator): Time estimator.
            Оценщик времени.
        sgs_type (ScheduleGenerationScheme): Schedule generation scheme.
            Схема генерации расписания.
    
    Returns:
        tuple[dict[GraphNode, ScheduledWork], Time, Timeline, list[GraphNode]]:
            Schedule data with start time, timeline and order.
            Данные расписания со временем начала, временной шкалой и порядком.
    ```

- **convert_schedule_to_chromosome**

    ```
    Transform schedule into chromosome.
    
    Преобразует расписание в хромосому.
    
    Args:
        work_id2index (dict[str, int]): Mapping of work IDs to indices.
            Отображение идентификаторов работ в индексы.
        worker_name2index (dict[str, int]): Mapping of worker types to indices.
            Отображение типов рабочих в индексы.
        contractor2index (dict[str, int]): Mapping of contractors to indices.
            Отображение подрядчиков в индексы.
        contractor_borders (np.ndarray): Capacity limits for contractors.
            Ограничения мощностей подрядчиков.
        schedule (Schedule): Source schedule.
            Исходное расписание.
        spec (ScheduleSpec): Scheduling specification.
            Спецификация расписания.
        landscape (LandscapeConfiguration): Landscape configuration.
            Конфигурация ландшафта.
        order (list[GraphNode] | None): Prescribed node order.
            Предопределенный порядок узлов.
    
    Returns:
        ChromosomeType: Generated chromosome.
            Сформированная хромосома.
    ```

- **parallel_schedule_generation_scheme**

    ```
    Convert chromosome using parallel scheme.
    
    Преобразует хромосому, используя параллельную схему.
    
    Returns:
        tuple[dict[GraphNode, ScheduledWork], Time, Timeline, list[GraphNode]]:
            Schedule data with start time, timeline and order.
            Данные расписания со временем начала, временной шкалой и порядком.
    ```

- **serial_schedule_generation_scheme**

    ```
    Convert chromosome using serial scheme.
    
    Преобразует хромосому, используя последовательную схему.
    
    Returns:
        tuple[dict[GraphNode, ScheduledWork], Time, Timeline, list[GraphNode]]:
            Schedule data with start time, timeline and order.
            Данные расписания со временем начала, временной шкалой и порядком.
    ```


### <a id="schedulergenetic-operatorspy"></a>[operators.py](scheduler/genetic/operators.py)

#### Classes / Классы
- **DeadlineCostFitness**

    ```
    Cost fitness with deadline constraint.
    
    Фитнес по стоимости с ограничением по дедлайну.
    ```

- **DeadlineResourcesFitness**

    ```
    Resource fitness with deadline constraint.
    
    Фитнес по ресурсам с ограничением по дедлайну.
    ```

- **SumOfResourcesFitness**

    ```
    Fitness from total resource usage.
    
    Фитнес по суммарному использованию ресурсов.
    ```

- **SumOfResourcesPeaksFitness**

    ```
    Fitness from sum of resource peaks.
    
    Фитнес по сумме пиков использования ресурсов.
    ```

- **TimeAndResourcesFitness**

    ```
    Bi-objective fitness of time and resource peaks.
    
    Двухцелевой фитнес по времени и пикам ресурсов.
    ```

- **TimeFitness**

    ```
    Fitness based on finish time.
    
    Фитнес, основанный на времени завершения.
    ```

- **TimeWithResourcesFitness**

    ```
    Fitness considering time and resource set.
    
    Фитнес с учетом времени и набора ресурсов.
    ```

#### Functions / Функции
- **copy_individual**

    ```
    Deep copy individual.
    
    Глубоко копирует индивида.
    
    Args:
        ind (Individual): Individual to copy.
            Индивид для копирования.
        toolbox (Toolbox): Genetic toolbox.
            Генетический набор инструментов.
    
    Returns:
        Individual: Copied individual.
            Скопированный индивид.
    ```

- **evaluate**

    ```
    Evaluate chromosome to schedule if valid.
    
    Оценивает хромосому в расписание, если она корректна.
    
    Args:
        chromosome (ChromosomeType): Chromosome to evaluate.
            Хромосома для оценки.
        wg (WorkGraph): Work graph.
            Граф работ.
        toolbox (Toolbox): Genetic toolbox.
            Генетический набор инструментов.
    
    Returns:
        Schedule | None: Built schedule or ``None`` when invalid.
            Построенное расписание или ``None`` при некорректности.
    ```

- **generate_chromosome**

    ```
    Generate a single valid chromosome.
    
    Генерирует одну валидную хромосому.
    
    Uses HEFT and randomized topological orders to respect dependencies.
    
    Использует порядок HEFT и случайные топологические сортировки для
    соблюдения зависимостей.
    ```

- **generate_chromosomes**

    ```
    Generate a list of chromosomes.
    
    Генерирует список хромосом.
    
    Args:
        n (int): Number of chromosomes.
            Количество хромосом.
        wg (WorkGraph): Work graph.
            Граф работ.
        contractors (list[Contractor]): Contractors list.
            Список подрядчиков.
        spec (ScheduleSpec): Scheduling specification.
            Спецификация расписания.
        work_id2index (dict[str, int]): Work ID to index map.
            Сопоставление идентификаторов работ индексам.
        worker_name2index (dict[str, int]): Worker name to index map.
            Сопоставление имен рабочих индексам.
        contractor2index (dict[str, int]): Contractor ID to index map.
            Сопоставление идентификаторов подрядчиков индексам.
        contractor_borders (np.ndarray): Contractor capacities.
            Вместимости подрядчиков.
        init_chromosomes (dict[str, tuple[ChromosomeType, float, ScheduleSpec]]):
            Predefined chromosomes with weights.
            Предопределенные хромосомы с весами.
        rand (random.Random): Random generator.
            Генератор случайных чисел.
        toolbox (Toolbox): Genetic toolbox.
            Генетический набор инструментов.
        work_estimator (WorkTimeEstimator | None): Time estimator.
            Оценщик времени.
        landscape (LandscapeConfiguration): Landscape configuration.
            Конфигурация ландшафта.
        only_lft_initialization (bool): Use only LFT initialization.
            Использовать только LFT-инициализацию.
    
    Returns:
        list[ChromosomeType]: Generated chromosomes.
            Сгенерированные хромосомы.
    ```

- **get_order_part**

    ```
    Extract new order fragment from second parent.
    
    Извлекает новый фрагмент порядка из второго родителя.
    ```

- **init_toolbox**

    ```
    Create toolbox with genetic operators.
    
    Создает набор инструментов с генетическими операторами.
    
    Returns:
        base.Toolbox: Configured toolbox for GA.
            Настроенный набор инструментов для ГА.
    ```

- **is_chromosome_contractors_correct**

    ```
    Ensure contractors can supply assigned workers.
    
    Убеждается, что подрядчики обеспечивают назначенных рабочих.
    ```

- **is_chromosome_correct**

    ```
    Check order and contractor borders for correctness.
    
    Проверяет корректность порядка работ и границ подрядчиков.
    ```

- **is_chromosome_order_correct**

    ```
    Verify that work order is topologically valid.
    
    Проверяет, что порядок работ топологически верен.
    ```

- **mate**

    ```
    Combined crossover for order, resources, and zones.
    
    Комбинированный кроссовер порядка, ресурсов и зон.
    
    Args:
        ind1 (Individual): First parent.
            Первый родитель.
        ind2 (Individual): Second parent.
            Второй родитель.
        optimize_resources (bool): Adjust borders after mating.
            Изменять границы после скрещивания.
        rand (random.Random): Random generator.
            Генератор случайных чисел.
        toolbox (Toolbox): Genetic toolbox.
            Генетический набор инструментов.
        priorities (np.ndarray): Node priorities.
            Приоритеты узлов.
    
    Returns:
        tuple[Individual, Individual]: Offspring individuals.
            Потомки.
    ```

- **mate_for_zones**

    ```
    One-point crossover for zones.
    
    Одноточечный кроссовер зон.
    
    Args:
        ind1 (Individual): First parent.
            Первый родитель.
        ind2 (Individual): Second parent.
            Второй родитель.
        rand (random.Random): Random generator.
            Генератор случайных чисел.
        toolbox (Toolbox): Genetic toolbox.
            Генетический набор инструментов.
        copy (bool): Copy individuals before mating.
            Копировать ли индивидов перед скрещиванием.
    
    Returns:
        tuple[Individual, Individual]: Offspring individuals.
            Потомки.
    ```

- **mate_resources**

    ```
    One-point crossover for resources.
    
    Одноточечный кроссовер ресурсов.
    
    Args:
        ind1 (Individual): First parent.
            Первый родитель.
        ind2 (Individual): Second parent.
            Второй родитель.
        optimize_resources (bool): Update resource borders after mating.
            Обновлять ли границы ресурсов после скрещивания.
        rand (random.Random): Random generator.
            Генератор случайных чисел.
        copy (bool): Copy individuals before mating.
            Копировать ли индивидов перед скрещиванием.
        toolbox (Toolbox): Genetic toolbox.
            Генетический набор инструментов.
    
    Returns:
        tuple[Individual, Individual]: Offspring individuals.
            Потомки.
    ```

- **mate_scheduling_order**

    ```
    Two-point crossover for work order.
    
    Двухточечный кроссовер порядка работ.
    
    Args:
        ind1 (Individual): First parent.
            Первый родитель.
        ind2 (Individual): Second parent.
            Второй родитель.
        rand (random.Random): Random generator.
            Генератор случайных чисел.
        toolbox (Toolbox): Genetic toolbox.
            Генетический набор инструментов.
        priorities (np.ndarray): Node priorities.
            Приоритеты узлов.
        copy (bool): Copy individuals before mating.
            Копировать индивидов перед скрещиванием.
    
    Returns:
        tuple[Individual, Individual]: Offspring individuals.
            Потомки.
    ```

- **mutate**

    ```
    Combined mutation for order, resources, and zones.
    
    Комбинированная мутация порядка, ресурсов и зон.
    
    Args:
        ind (Individual): Individual to mutate.
            Индивид для мутации.
        resources_border (np.ndarray): Resource borders.
            Границы ресурсов.
        parents (dict[int, set[int]]): Parents mapping.
            Отображение родителей.
        children (dict[int, set[int]]): Children mapping.
            Отображение потомков.
        statuses_available (int): Number of statuses.
            Количество статусов.
        order_mutpb (float): Order mutation probability.
            Вероятность мутации порядка.
        res_mutpb (float): Resource mutation probability.
            Вероятность мутации ресурсов.
        zone_mutpb (float): Zone mutation probability.
            Вероятность мутации зон.
        rand (random.Random): Random generator.
            Генератор случайных чисел.
    
    Returns:
        Individual: Mutated individual.
            Мутированный индивид.
    ```

- **mutate_for_zones**

    ```
    Mutate zone statuses of an individual.
    
    Мутирует статус зон у индивида.
    
    Args:
        ind (Individual): Individual to mutate.
            Индивид для мутации.
        mutpb (float): Mutation probability.
            Вероятность мутации.
        rand (random.Random): Random generator.
            Генератор случайных чисел.
        statuses_available (int): Number of available statuses.
            Количество доступных статусов.
    
    Returns:
        Individual: Mutated individual.
            Мутированный индивид.
    ```

- **mutate_resource_borders**

    ```
    Mutate contractors' resource borders.
    
    Мутирует границы ресурсов подрядчиков.
    
    Args:
        ind (Individual): Individual to mutate.
            Индивид для мутации.
        mutpb (float): Mutation probability.
            Вероятность мутации.
        rand (random.Random): Random generator.
            Генератор случайных чисел.
        contractor_borders (np.ndarray): Upper capacity borders.
            Верхние границы мощностей.
    
    Returns:
        Individual: Mutated individual.
            Мутированный индивид.
    ```

- **mutate_resources**

    ```
    Mutate resources of an individual.
    
    Мутирует ресурсы индивида.
    
    Args:
        ind (Individual): Individual to mutate.
            Индивид для мутации.
        mutpb (float): Mutation probability.
            Вероятность мутации.
        rand (random.Random): Random generator.
            Генератор случайных чисел.
        resources_border (np.ndarray): Lower and upper resource borders.
            Нижние и верхние границы ресурсов.
    
    Returns:
        Individual: Mutated individual.
            Мутированный индивид.
    ```

- **mutate_scheduling_order**

    ```
    Mutate work order of an individual.
    
    Мутирует порядок работ индивида.
    
    Args:
        ind (Individual): Individual to mutate.
            Индивид для мутации.
        mutpb (float): Mutation probability.
            Вероятность мутации.
        rand (random.Random): Random generator.
            Генератор случайных чисел.
        priorities (np.ndarray): Node priorities.
            Приоритеты узлов.
        parents (dict[int, set[int]]): Parent mapping for order validity.
            Отображение родителей для валидности порядка.
        children (dict[int, set[int]]): Children mapping for order validity.
            Отображение потомков для валидности порядка.
    
    Returns:
        Individual: Mutated individual.
            Мутированный индивид.
    ```

- **mutate_scheduling_order_core**

    ```
    Core mutation for work order respecting dependencies.
    
    Ядро мутации порядка работ с учетом зависимостей.
    
    Args:
        order (np.ndarray): Current order.
            Текущий порядок.
        mutpb (float): Mutation probability.
            Вероятность мутации.
        rand (random.Random): Random generator.
            Генератор случайных чисел.
        parents (dict[int, set[int]]): Parent mapping.
            Отображение родителей.
        children (dict[int, set[int]]): Children mapping.
            Отображение потомков.
    ```

- **mutate_values**

    ```
    Change numeric values in chromosome slice.
    
    Изменяет числовые значения в части хромосомы.
    ```

- **register_individual_constructor**

    ```
    Register individual type with custom fitness.
    
    Регистрирует тип индивида с пользовательским фитнесом.
    
    Args:
        fitness_weights (tuple[int | float, ...]): Fitness weights.
            Веса функции пригодности.
        toolbox (base.Toolbox): Target toolbox.
            Целевой набор инструментов.
    ```

- **select_new_population**

    ```
    Select top individuals by fitness.
    
    Выбирает лучших индивидов по пригодности.
    
    Args:
        population (list[Individual]): Population to select from.
            Популяция для отбора.
        k (int): Number of individuals to select.
            Количество выбираемых индивидов.
    
    Returns:
        list[Individual]: Selected individuals.
            Отобранные индивиды.
    ```

- **two_point_order_crossover**

    ```
    Perform two-point crossover on order chromosome.
    
    Выполняет двухточечный кроссовер на хромосоме порядка.
    
    Args:
        child (np.ndarray): Order of first parent.
            Порядок первого родителя.
        other_parent (np.ndarray): Order of second parent.
            Порядок второго родителя.
        min_mating_amount (int): Minimum crossover segment.
            Минимальный сегмент кроссовера.
        rand (random.Random): Random generator.
            Генератор случайных чисел.
    
    Returns:
        np.ndarray: Updated order.
            Обновленный порядок.
    ```


### <a id="schedulergenetic-schedule_builderpy"></a>[schedule_builder.py](scheduler/genetic/schedule_builder.py)

#### Functions / Функции
- **build_schedules**

    ```
    Build schedules using a genetic algorithm.
    
    Строит расписания с использованием генетического алгоритма.
    ```

- **build_schedules_with_cache**

    ```
    Run genetic algorithm returning schedules and chromosomes.
    
    Запускает генетический алгоритм, возвращая расписания и хромосомы.
    
    Returns:
        tuple[list[tuple[ScheduleWorkDict, Time, Timeline, list[GraphNode]]], list[ChromosomeType]]:
            Generated schedules and final population.
            Сгенерированные расписания и итоговая популяция.
    ```

- **compare_individuals**

- **create_toolbox**

    ```
    Prepare toolbox for genetic scheduling.
    
    Подготавливает набор инструментов для генетического планирования.
    
    Returns:
        Toolbox: Configured toolbox.
            Настроенный набор инструментов.
    ```

- **make_offspring**


### <a id="schedulergenetic-utilspy"></a>[utils.py](scheduler/genetic/utils.py)

#### Functions / Функции
- **create_toolbox_using_cached_chromosomes**

    ```
    Create toolbox reusing cached chromosomes.
    
    Создает набор инструментов, используя закэшированные хромосомы.
    ```

- **init_chromosomes_f**

    ```
    Convert initial schedules to chromosomes.
    
    Преобразует начальные расписания в хромосомы.
    ```

- **prepare_optimized_data_structures**

    ```
    Prepare data structures for fast access.
    
    Подготавливает структуры данных для быстрого доступа.
    ```


## <a id="schedulerheft"></a>scheduler/heft

### <a id="schedulerheft-basepy"></a>[base.py](scheduler/heft/base.py)

#### Classes / Классы
- **HEFTBetweenScheduler**

    ```
    Type of scheduler that use method of critical path.
    The scheduler give opportunity to add work between existing works.
    ```

- **HEFTScheduler**

    ```
    Scheduler that uses method of a critical path.
    The scheduler gives opportunity to add work only to end.
    ```


### <a id="schedulerheft-prioritizationpy"></a>[prioritization.py](scheduler/heft/prioritization.py)

#### Functions / Функции
- **ford_bellman**

    ```
    Runs heuristic ford-bellman algorithm for given graph and weights.
    ```

- **prioritization**

    ```
    Return ordered critical nodes.
    Finish time is depended on these ordered nodes.
    ```

- **prioritization_nodes**

    ```
    Return ordered critical nodes.
    Finish time is depended on these ordered nodes.
    ```


## <a id="schedulerlft"></a>scheduler/lft

### <a id="schedulerlft-basepy"></a>[base.py](scheduler/lft/base.py)

#### Classes / Классы
- **LFTScheduler**

    ```
    Schedule works using the MIN-LFT priority rule.
    
    Планирует работы с использованием правила приоритета MIN-LFT.
    ```

- **RandomizedLFTScheduler**

    ```
    Stochastic version of ``LFTScheduler``.
    
    Стохастический вариант ``LFTScheduler``.
    ```

#### Functions / Функции
- **get_contractors_and_workers_amounts_for_work**

    ```
    Select suitable contractors and assign worker amounts.
    
    Выбрать подходящих подрядчиков и назначить количество рабочих.
    
    Args:
        work_unit (WorkUnit): Work to be performed.
            Работа, которую необходимо выполнить.
        contractors (list[Contractor]): Available contractors.
            Доступные подрядчики.
        spec (ScheduleSpec): Scheduling constraints.
            Ограничения планирования.
        worker_pool (WorkerContractorPool): Available workforce per contractor.
            Доступная рабочая сила у подрядчиков.
    
    Returns:
        tuple[list[Contractor], np.ndarray]: Contractors and worker amounts.
            Подрядчики и количество рабочих.
    
    Raises:
        IncorrectAmountOfWorker: Assigned worker counts are invalid.
            Назначенное количество рабочих неверно.
        NoSufficientContractorError: No contractor satisfies requirements.
            Ни один подрядчик не удовлетворяет требованиям.
    ```


### <a id="schedulerlft-prioritizationpy"></a>[prioritization.py](scheduler/lft/prioritization.py)

#### Functions / Функции
- **lft_prioritization**

    ```
    Order critical nodes by a core prioritization function.
    
    Упорядочить критические узлы с помощью базовой функции приоритезации.
    
    Args:
        head_nodes (list[GraphNode]): Nodes to order.
            Узлы для упорядочивания.
        node_id2parent_ids (dict[str, set[str]]): Mapping of parent IDs.
            Отображение идентификаторов родителей.
        node_id2child_ids (dict[str, set[str]]): Mapping of child IDs.
            Отображение идентификаторов детей.
        node_id2duration (dict[str, int]): Durations for each node.
            Длительности для каждого узла.
        core_f (Callable): Core prioritization function.
            Базовая функция приоритезации.
        rand (random.Random, optional): Random generator.
            Генератор случайных чисел.
    
    Returns:
        list[GraphNode]: Ordered nodes.
            Упорядоченные узлы.
    ```

- **lft_prioritization_core**

    ```
    Order nodes by the MIN-LFT priority rule.
    
    Упорядочить узлы по правилу приоритета MIN-LFT.
    
    Args:
        head_nodes (list[GraphNode]): Nodes to order.
            Узлы для упорядочивания.
        node_id2parent_ids (dict[str, set[str]]): Mapping of parent IDs.
            Отображение идентификаторов родителей.
        node_id2child_ids (dict[str, set[str]]): Mapping of child IDs.
            Отображение идентификаторов детей.
        node_id2duration (dict[str, int]): Durations for each node.
            Длительности для каждого узла.
        groups (list[list[GraphNode]]): Priority groups.
            Группы приоритетов.
        rand (random.Random, optional): Random generator.
            Генератор случайных чисел.
    
    Returns:
        list[GraphNode]: Ordered nodes.
            Упорядоченные узлы.
    ```

- **lft_randomized_prioritization_core**

    ```
    Sample nodes using MIN-LFT and MIN-LST rules.
    
    Отобрать узлы с использованием правил MIN-LFT и MIN-LST.
    
    Args:
        head_nodes (list[GraphNode]): Nodes to order.
            Узлы для упорядочивания.
        node_id2parent_ids (dict[str, set[str]]): Mapping of parent IDs.
            Отображение идентификаторов родителей.
        node_id2child_ids (dict[str, set[str]]): Mapping of child IDs.
            Отображение идентификаторов детей.
        node_id2duration (dict[str, int]): Durations for each node.
            Длительности для каждого узла.
        groups (list[list[GraphNode]]): Priority groups.
            Группы приоритетов.
        rand (random.Random, optional): Random generator.
            Генератор случайных чисел.
    
    Returns:
        list[GraphNode]: Ordered nodes.
            Упорядоченные узлы.
    ```

- **map_lft_lst**

    ```
    Map nodes to LFT and LST values.
    
    Сопоставить узлы значениям LFT и LST.
    
    Args:
        head_nodes (list[GraphNode]): Nodes in topological order.
            Узлы в топологическом порядке.
        node_id2child_ids (dict[str, set[str]]): Mapping of node IDs to child IDs.
            Отображение идентификаторов узлов на идентификаторы их детей.
        node_id2duration (dict[str, int]): Estimated durations.
            Оценённые длительности.
    
    Returns:
        tuple[dict[str, int], dict[str, int]]: Dictionaries of LFT and LST values.
            Словари значений LFT и LST.
    ```


### <a id="schedulerlft-time_computaionpy"></a>[time_computaion.py](scheduler/lft/time_computaion.py)

#### Functions / Функции
- **get_chain_duration**


## <a id="schedulermulti_agency"></a>scheduler/multi_agency

### <a id="schedulermulti_agency-block_generatorpy"></a>[block_generator.py](scheduler/multi_agency/block_generator.py)

#### Classes / Классы
- **SyntheticBlockGraphType**

    ```
    Types of synthetic block graphs.
    
    Типы синтетических графов блоков.
    
    Attributes:
        SEQUENTIAL: Works are performed mostly sequentially.
            Работы выполняются преимущественно последовательно.
        PARALLEL: Works can be performed mostly in parallel.
            Работы могут выполняться преимущественно параллельно.
        RANDOM: Random structure of a block graph.
            Случайная структура графа блоков.
        QUEUES: Queue structure typical of real construction processes.
            Очередная структура, типичная для процессов капитального строительства.
    ```

#### Functions / Функции
- **generate_block_graph**

    ```
    Generate a block graph of the given type.
    
    Сгенерировать граф блоков заданного типа.
    
    Args:
        graph_type: Desired structure of the block graph.
            Требуемая структура графа блоков.
        n_blocks: Number of blocks.
            Количество блоков.
        type_prop: Proportions of `WorkGraph` types: General, Parallel, Sequential, Queues.
            Пропорции типов `WorkGraph`: общий, параллельный, последовательный, очереди.
        count_supplier: Function returning size limits for a block index.
            Функция, возвращающая границы размера для индекса блока.
        edge_prob: Probability that an edge exists.
            Вероятность существования ребра.
        rand: Randomness source.
            Источник случайности.
        obstruction_getter: Function providing an optional obstruction for a block.
            Функция, предоставляющая необязательное препятствие для блока.
        queues_num: Number of queues in the block graph.
            Количество очередей в графе блоков.
        queues_blocks: Number of blocks in each queue.
            Количество блоков в каждой очереди.
        queues_edges: Number of edges in each queue.
            Количество рёбер в каждой очереди.
        logger: Log function consuming messages.
            Функция логирования, принимающая сообщения.
    
    Returns:
        BlockGraph: Generated block graph.
            Сгенерированный граф блоков.
    ```

- **generate_blocks**

    ```
    Generate a synthetic block graph.
    
    Сгенерировать синтетический граф блоков.
    
    Args:
        graph_type: Type of the resulting block graph.
            Тип результирующего графа блоков.
        n_blocks: Number of blocks.
            Количество блоков.
        type_prop: Proportions of `WorkGraph` types: General, Parallel, Sequential.
            Пропорции типов `WorkGraph`: общий, параллельный, последовательный.
        count_supplier: Function that returns size limits for a block index.
            Функция, возвращающая границы размера для индекса блока.
        edge_prob: Probability that an edge exists.
            Вероятность существования ребра.
        rand: Randomness source.
            Источник случайности.
        obstruction_getter: Function providing an optional obstruction for a block.
            Функция, предоставляющая необязательное препятствие для блока.
        logger: Log function consuming messages.
            Функция логирования, принимающая сообщения.
    
    Returns:
        BlockGraph: Generated block graph.
            Сгенерированный граф блоков.
    ```

- **generate_empty_graph**

    ```
    Create a minimal work graph with start and end nodes.
    
    Создать минимальный граф работ с начальными и конечными вершинами.
    
    Returns:
        WorkGraph: Generated empty graph.
            Сгенерированный пустой граф.
    ```

- **generate_queues**

    ```
    Generate a block graph with queue structure.
    
    Сгенерировать граф блоков с очередной структурой.
    
    Args:
        type_prop: Proportions of `WorkGraph` types: General, Parallel, Sequential.
            Пропорции типов `WorkGraph`: общий, параллельный, последовательный.
        count_supplier: Function returning size limits for a block index.
            Функция, возвращающая границы размера для индекса блока.
        rand: Randomness source.
            Источник случайности.
        obstruction_getter: Function providing an optional obstruction for a block.
            Функция, предоставляющая необязательное препятствие для блока.
        queues_num: Number of queues in the block graph.
            Количество очередей в графе блоков.
        queues_blocks: Number of blocks in each queue.
            Количество блоков в каждой очереди.
        queues_edges: Number of edges in each queue.
            Количество рёбер в каждой очереди.
        logger: Log function consuming messages.
            Функция логирования, принимающая сообщения.
    
    Returns:
        BlockGraph: Generated block graph.
            Сгенерированный граф блоков.
    ```


### <a id="schedulermulti_agency-block_graphpy"></a>[block_graph.py](scheduler/multi_agency/block_graph.py)

#### Classes / Классы
- **BlockGraph**

    ```
    Graph composed of work blocks connected by finish-start edges.
    
    Граф, составленный из блоков работ, связанных зависимостями типа
    "конец-начало".
    ```

- **BlockNode**

    ```
    Node of a block graph with its work graph and relations.
    
    Узел графа блоков с его графом работ и связями.
    
    Attributes:
        wg: Work graph contained in the node.
            Рабочий граф, содержащийся в узле.
        obstruction: Optional obstruction for the block.
            Необязательное препятствие для блока.
        blocks_from: Predecessor block nodes.
            Узлы-предшественники.
        blocks_to: Successor block nodes.
            Узлы-потомки.
    ```


### <a id="schedulermulti_agency-block_validationpy"></a>[block_validation.py](scheduler/multi_agency/block_validation.py)

#### Functions / Функции
- **_check_block_dependencies**

    ```
    Validate right block dependencies considering received 'schedule'
    
    :param bg: BlockGraph
    :param schedule:
    ```

- **_check_blocks_separately**

    ```
    Validate each block separately, i.e., check each 'Schedule' in each block
    
    :param sblocks: scheduled blocks of works
    ```

- **_check_blocks_with_global_timelines**

    ```
    Checks that no agent's contractor uses more resources that can supply.
    
    Note that this should fail if there is a shared contractor between agents, but this
    term is, of course, unsupported.
    
    :param sblocks: scheduled blocks of works
    :param contractors: global scope of contractors(collected from all agents used to construct sblocks)
    ```

- **validate_block_schedule**


### <a id="schedulermulti_agency-exceptionpy"></a>[exception.py](scheduler/multi_agency/exception.py)

#### Classes / Классы
- **NoSufficientAgents**

    ```
    Raise when manager does not have enough agents
    ```


### <a id="schedulermulti_agency-multi_agencypy"></a>[multi_agency.py](scheduler/multi_agency/multi_agency.py)

#### Classes / Классы
- **Agent**

    ```
    Represents an agent capable of bidding on blocks.
    
    Представляет агента, способного делать ставки на блоки.
    ```

- **Manager**

    ```
    Manager that orchestrates agents.
    
    Менеджер, который координирует агентов.
    ```

- **NeuralManager**

    ```
    Manager that selects agents using neural networks.
    
    Менеджер, выбирающий агентов с помощью нейронных сетей.
    
    Args:
        agents: Agents each with its scheduler and contractors.
            Агенты, каждый со своим планировщиком и подрядчиками.
        algo_trainer: Neural network predicting the best scheduling algorithm.
            Нейросеть, предсказывающая наилучший алгоритм планирования.
        contractor_trainer: Neural network predicting contractor resources.
            Нейросеть, предсказывающая ресурсы подрядчиков.
        algorithms: List of unique schedulers used by agents.
            Список уникальных планировщиков, используемых агентами.
        blocks: Blocks of the input block graph in topological order.
            Блоки входного графа блоков в топологическом порядке.
        encoding_blocks: Embeddings of graph blocks.
            Векторные представления блоков графа.
    ```

- **ScheduledBlock**

    ```
    Result of scheduling a block of works.
    
    Результат планирования блока работ.
    
    Attributes:
        wg: Scheduled work graph.
            Запланированный граф работ.
        schedule: Schedule produced for the block.
            График, построенный для блока.
        agent: Agent that scheduled the block.
            Агент, который запланировал блок.
        start_time: Global start time of the block.
            Глобальное время начала блока.
        end_time: Global end time of the block.
            Глобальное время окончания блока.
    ```

- **StochasticManager**

    ```
    Manager using confidence levels to adjust agent offers.
    
    Менеджер, использующий уровни доверия для коррекции предложений агентов.
    ```


## <a id="schedulerresource"></a>scheduler/resource

### <a id="schedulerresource-average_reqpy"></a>[average_req.py](scheduler/resource/average_req.py)

#### Classes / Классы
- **AverageReqResourceOptimizer**

    ```
    Class that implements optimization the number of resources by counting average resource requirements.
    ```


### <a id="schedulerresource-basepy"></a>[base.py](scheduler/resource/base.py)

#### Classes / Классы
- **ResourceOptimizer**

    ```
    Base class to build different methods of resource optimization.
    Constructed methods minimize the quantity of resources.
    ```


### <a id="schedulerresource-coordinate_descentpy"></a>[coordinate_descent.py](scheduler/resource/coordinate_descent.py)

#### Classes / Классы
- **CoordinateDescentResourceOptimizer**

    ```
    Class that implements optimization the number of resources by discrete analogue of coordinate descent.
    ```


### <a id="schedulerresource-full_scanpy"></a>[full_scan.py](scheduler/resource/full_scan.py)

#### Classes / Классы
- **FullScanResourceOptimizer**

    ```
    Class that implements optimization the number of resources by the smart search method.
    ```


### <a id="schedulerresource-identitypy"></a>[identity.py](scheduler/resource/identity.py)

#### Classes / Классы
- **IdentityResourceOptimizer**

    ```
    Empty class of resource optimizer.
    ```


## <a id="schedulerresources_in_time"></a>scheduler/resources_in_time

### <a id="schedulerresources_in_time-average_binary_searchpy"></a>[average_binary_search.py](scheduler/resources_in_time/average_binary_search.py)

#### Classes / Классы
- **AverageBinarySearchResourceOptimizingScheduler**

    ```
    Optimize resource multiplier using binary search.
    
    Оптимизирует множитель ресурсов посредством двоичного поиска.
    ```


## <a id="schedulerselection"></a>scheduler/selection

### <a id="schedulerselection-metricspy"></a>[metrics.py](scheduler/selection/metrics.py)

#### Functions / Функции
- **encode_graph**

    ```
    Encode graph structure into feature vector.
    
    Кодирует структуру графа в вектор признаков.
    
    Args:
        wg: Work graph to encode.
            Граф работ для кодирования.
    
    Returns:
        List of graph features.
        Список признаков графа.
    ```

- **metric_average_resource_usage**

    ```
    Average number of requested workers per node.
    
    Среднее число требуемых работников на узел.
    
    Args:
        wg: Work graph.
            Граф работ.
    
    Returns:
        Average resource usage.
        Среднее использование ресурсов.
    ```

- **metric_average_work_per_activity**

    ```
    Average work volume per node.
    
    Средний объём работы на узел.
    
    Args:
        wg: Work graph.
            Граф работ.
    
    Returns:
        Average work volume.
        Средний объём работы.
    ```

- **metric_graph_parallelism_degree**

    ```
    Estimate degree of parallel execution per graph level.
    
    Оценивает степень параллельного выполнения по уровням графа.
    
    Args:
        wg: Work graph to analyze.
            Граф работ для анализа.
    
    Returns:
        Averaged parallelism degrees for batches.
        Усреднённые степени параллельности по батчам.
    ```

- **metric_longest_path**

    ```
    Compute length of the longest path in graph.
    
    Вычисляет длину самого длинного пути в графе.
    
    Args:
        wg: Work graph to analyze.
            Граф работ для анализа.
    
    Returns:
        Length of the longest path.
        Длина самого длинного пути.
    ```

- **metric_relative_max_children**

    ```
    Relative maximum number of children for a node.
    
    Относительное максимальное число потомков для узла.
    
    Args:
        wg: Work graph.
            Граф работ.
    
    Returns:
        Ratio of maximum children to total vertices.
        Отношение максимального числа потомков к числу вершин.
    ```

- **metric_relative_max_parents**

    ```
    Relative maximum number of parents for a node.
    
    Относительное максимальное число родителей для узла.
    
    Args:
        wg: Work graph.
            Граф работ.
    
    Returns:
        Ratio of maximum parents to total vertices.
        Отношение максимального числа родителей к числу вершин.
    ```

- **metric_resource_constrainedness**

    ```
    Calculate constrainedness for each resource type.
    
    Вычисляет степень ограниченности для каждого типа ресурса.
    
    The constrainedness equals the average requested units divided by the
    capacity of the resource.
    
    Ограниченность равна среднему количеству запрошенных единиц, делённому
    на вместимость ресурса.
    
    Args:
        wg: Work graph to analyze.
            Граф работ для анализа.
    
    Returns:
        List of constrainedness coefficients.
        Список коэффициентов ограниченности.
    ```

- **metric_vertex_count**

    ```
    Return number of vertices in graph.
    
    Возвращает число вершин в графе.
    
    Args:
        wg: Work graph.
            Граф работ.
    
    Returns:
        Vertex count.
        Количество вершин.
    ```

- **one_hot_decode**

    ```
    Decode one-hot tensor back to index.
    
    Декодирует one-hot тензор обратно в индекс.
    
    Args:
        v: Tensor to decode.
            Тензор для декодирования.
    
    Returns:
        Decoded index.
        Декодированный индекс.
    ```

- **one_hot_encode**

    ```
    Convert index into one-hot vector.
    
    Преобразует индекс в one-hot вектор.
    
    Args:
        v: Index to encode.
            Индекс для кодирования.
        max_v: Length of result vector.
            Длина результирующего вектора.
    
    Returns:
        One-hot encoded list.
        Список в формате one-hot.
    ```


### <a id="schedulerselection-neural_netpy"></a>[neural_net.py](scheduler/selection/neural_net.py)

#### Classes / Классы
- **NeuralNet**

    ```
    Feedforward neural network for scheduling metrics.
    
    Полносвязная нейросеть для работы с метриками расписаний.
    ```

- **NeuralNetTrainer**

    ```
    Utility class for training and evaluation.
    
    Вспомогательный класс для обучения и оценки.
    ```

- **NeuralNetType**

    ```
    Enumeration of supported neural net task types.
    
    Перечисление поддерживаемых типов задач для нейросети.
    ```


### <a id="schedulerselection-validationpy"></a>[validation.py](scheduler/selection/validation.py)

#### Functions / Функции
- **cross_val_score**

    ```
    Evaluate metric by cross-validation and also record score times.
    
    :param X: The data to fit (DataFrame).
    :param y: The column that contains target variable to try to predict.
    :param model: The object (inherited from nn.Module).
    :param epochs: Number of epochs during which the model is trained.
    :param folds: Training dataset is split on 'folds' folds for cross-validation.
    :param shuffle: 'True' if the splitting dataset on folds should be random, 'False' - otherwise.
    :param random_state:
    :return: List of scores that correspond to each validation fold.
    ```


## <a id="schedulertimeline"></a>scheduler/timeline

### <a id="schedulertimeline-basepy"></a>[base.py](scheduler/timeline/base.py)

#### Classes / Классы
- **BaseSupplyTimeline**

- **Timeline**

    ```
    Entity that saves info on the use of resources over time.
    Timeline provides opportunities to work with GraphNodes and resources over time.
    ```


### <a id="schedulertimeline-general_timelinepy"></a>[general_timeline.py](scheduler/timeline/general_timeline.py)

#### Classes / Классы
- **GeneralTimeline**

    ```
    The representation of general-purpose timeline that supports some general subset of functions
    ```


### <a id="schedulertimeline-hybrid_supply_timelinepy"></a>[hybrid_supply_timeline.py](scheduler/timeline/hybrid_supply_timeline.py)

#### Classes / Классы
- **HybridSupplyTimeline**

    ```
    Material Timeline that implements the hybrid approach of resource supply -
    compares the time of resource delivery to work start and the time of delivery starting from the work start
    ```


### <a id="schedulertimeline-just_in_time_timelinepy"></a>[just_in_time_timeline.py](scheduler/timeline/just_in_time_timeline.py)

#### Classes / Классы
- **JustInTimeTimeline**

    ```
    Timeline that stored the time of resources release.
    For each contractor and worker type store a descending list of pairs of time and
    number of available workers of this type of this contractor.
    ```


### <a id="schedulertimeline-momentum_timelinepy"></a>[momentum_timeline.py](scheduler/timeline/momentum_timeline.py)

#### Classes / Классы
- **MomentumTimeline**

    ```
    Timeline that stores the intervals in which resources is occupied.
    ```


### <a id="schedulertimeline-platform_timelinepy"></a>[platform_timeline.py](scheduler/timeline/platform_timeline.py)

#### Classes / Классы
- **PlatformTimeline**


### <a id="schedulertimeline-to_start_supply_timelinepy"></a>[to_start_supply_timeline.py](scheduler/timeline/to_start_supply_timeline.py)

#### Classes / Классы
- **ToStartSupplyTimeline**


### <a id="schedulertimeline-utilspy"></a>[utils.py](scheduler/timeline/utils.py)

#### Functions / Функции
- **get_exec_times_from_assigned_time_for_chain**

    ```
    Distributes a given total execution time among work nodes in an inseparable chain.
    
    The time distribution is proportional to each node's volume, ensuring that
    the entire `assigned_time` is utilized. Any rounding discrepancies are
    allocated to the last node in the chain.
    
    Args:
        inseparable_chain: A list of nodes representing an inseparable sequence of work units.
        assigned_time: The total `Time` allocated for the entire chain's execution.
    
    Returns:
        A dictionary mapping each `GraphNode` to a tuple `(lag, node_execution_time)`.
        `lag` is always `Time(0)` as the chain is inseparable, and
        `node_execution_time` is the calculated execution time for that specific node.
    ```


### <a id="schedulertimeline-zone_timelinepy"></a>[zone_timeline.py](scheduler/timeline/zone_timeline.py)

#### Classes / Классы
- **ZoneTimeline**


## <a id="schedulertopological"></a>scheduler/topological

### <a id="schedulertopological-basepy"></a>[base.py](scheduler/topological/base.py)

#### Classes / Классы
- **RandomizedTopologicalScheduler**

    ```
    Topological scheduler with random tie-breaking.
    Топологический планировщик со случайным разрешением связей.
    ```

- **TopologicalScheduler**

    ```
    Scheduler representing a work graph in topological order.
    Планировщик, представляющий граф работ в топологическом порядке.
    ```

#### Functions / Функции
- **toposort**

    ```
    Perform topological sort on dependency mapping.
    Выполняет топологическую сортировку по отображению зависимостей.
    
    Args:
        data: Mapping of items to their dependencies.
            Отображение элементов и их зависимостей.
    
    Yields:
        set[str]: Sets of items in topological order.
            set[str]: Наборы элементов в топологическом порядке.
    ```

- **validate_order**

    ```
    Validate that order respects dependencies.
    Проверяет, что порядок учитывает зависимости.
    
    Args:
        order: Ordered list of graph nodes.
            Упорядоченный список узлов графа.
    ```


## <a id="schedulerutils"></a>scheduler/utils

### <a id="schedulerutils-__init__py"></a>[__init__.py](scheduler/utils/__init__.py)

#### Functions / Функции
- **get_head_nodes_with_connections_mappings**

    ```
    Identifies 'head nodes' in a WorkGraph and reconstructs their inter-node dependencies.
    
    Head nodes are defined as the first nodes of inseparable chains or standalone nodes
    that are not part of an inseparable chain (i.e., they are not 'inseparable sons').
    This function effectively flattens the graph by treating inseparable chains as
    single logical entities represented by their head node, and then re-establishes
    parent-child relationships between these head nodes.
    
    Args:
        wg: The `WorkGraph` to analyze.
    
    Returns:
        A tuple containing:
            - A list of `GraphNode` objects representing the head nodes,
              sorted in topological order based on their reconstructed dependencies.
            - A dictionary mapping the ID of each head node to a set of IDs of
              its new 'parent' head nodes. These represent external dependencies
              where a parent of any node within the current head node's inseparable
              chain belongs to another head node's chain.
            - A dictionary mapping the ID of each head node to a set of IDs of
              its new 'child' head nodes. Similar to parents, these represent
              external dependencies where a child of any node within the current
              head node's inseparable chain belongs to another head node's chain.
    ```

- **get_worker_contractor_pool**

    ```
    Gets worker-contractor dictionary from contractors list.
    Alias for frequently used functionality.
    
    :param contractors: list of all the considered contractors
    :return: dictionary of workers by worker name, next by contractor id
    ```


### <a id="schedulerutils-local_optimizationpy"></a>[local_optimization.py](scheduler/utils/local_optimization.py)

#### Classes / Классы
- **OrderLocalOptimizer**

    ```
    Base interface for node order optimizers.
    
    Базовый интерфейс оптимизаторов порядка узлов.
    ```

- **ParallelizeScheduleLocalOptimizer**

    ```
    Make nearby works execute in parallel.
    
    Заставляет близкие работы выполняться параллельно.
    ```

- **ScheduleLocalOptimizer**

    ```
    Base class for schedule-level local optimization.
    
    Базовый класс для локальной оптимизации расписаний.
    ```

- **SwapOrderLocalOptimizer**

    ```
    Shuffle nodes without violating topological order.
    
    Переставляет узлы, не нарушая топологического порядка.
    ```

#### Functions / Функции
- **get_swap_candidates**

    ```
    Find nodes swappable with target without breaking order.
    
    Находит узлы, которые можно обменять с целевым, не нарушая порядок.
    
    Args:
        node: Target node.
            Целевой узел.
        node_index: Index of target node in sequence.
            Индекс целевого узла в последовательности.
        candidates: Iterable of nodes to try.
            Перечень кандидатов для обмена.
        node2ind: Mapping from node to its index.
            Отображение узла в индекс.
        processed: Set of nodes to skip.
            Множество узлов, которые нужно пропустить.
    
    Returns:
        List of acceptable swap candidates.
        Список подходящих кандидатов для обмена.
    ```

- **optimize_local_sequence**

    ```
    Experimental local sequence optimizer.
    
    Экспериментальный оптимизатор локальной последовательности.
    
    Args:
        seq: Sequence of nodes.
            Последовательность узлов.
        start_ind: Start index.
            Начальный индекс.
        end_ind: End index.
            Конечный индекс.
        work_estimator: Work time estimator.
            Оценщик времени работы.
    
    TODO: Try to find sets of works with similar resources and parallelize.
    TODO: Попробовать находить работы с похожими ресурсами для параллелизации.
    ```


### <a id="schedulerutils-multi_contractorpy"></a>[multi_contractor.py](scheduler/utils/multi_contractor.py)

#### Functions / Функции
- **get_worker_borders**

    ```
    Define for each job each type of workers the min and max possible number of workers.
    For max number of workers, max is defined as a minimum from max possible numbers
    at all and max possible for a current job.
    
    :param agents: from all projects
    :param contractor:
    :param work_reqs:
    :return:
    ```

- **run_contractor_search**

    ```
    Performs the best contractor search.
    
    :param contractors: contractors' list
    :param runner: a runner function, should be inner of the calling code.
        Calculates Tuple[start time, finish time, worker team] from given contractor object.
    :return: start time, finish time, the best contractor, worker team with the best contractor
    ```


### <a id="schedulerutils-obstructionpy"></a>[obstruction.py](scheduler/utils/obstruction.py)

#### Classes / Классы
- **Obstruction**

    ```
    Tests the probability and, if it's true, apply the obstruction.
    ```

- **OneInsertObstruction**

    ```
    Applying seeks the random part of given WorkGraph and inserts it into that point.
    ```


### <a id="schedulerutils-time_computaionpy"></a>[time_computaion.py](scheduler/utils/time_computaion.py)

#### Functions / Функции
- **calculate_working_time**

    ```
    Calculate the working time of the appointed workers at a current job for final schedule
    
    :return: working time
    ```

- **calculate_working_time_cascade**

    ```
    Calculate the working time of the appointed workers at a current job for prioritization.
    O(1) - at worst case |inseparable_edges|
    
    :param appointed_worker:
    :param work_estimator:
    :param node: the target node
    :return: working time
    ```

- **work_priority**

    ```
    Calculate the average time to complete the work when assigning the minimum and maximum number of employees
    for the correct calculations of rank in prioritization
    O(sum_of_max_counts_of_workers) of current work
    
    :param node: the target node
    :param work_estimator:
    :param comp_cost: function for calculating working time (calculate_working_time)
    :return: average working time
    ```


## <a id="schemas"></a>schemas

### <a id="schemas-apply_queuepy"></a>[apply_queue.py](schemas/apply_queue.py)

#### Classes / Классы
- **ApplyQueue**

    ```
    Class represents the function apply sequence
    ```


### <a id="schemas-contractorpy"></a>[contractor.py](schemas/contractor.py)

#### Classes / Классы
- **Contractor**

    ```
    Used to store information about the contractor and its resources
    :param workers: dictionary, where the key is the employee's specialty, and the value is the pool of employees of
    this specialty
    :param equipments: dictionary, where the key is the type of technique, and the value is the pool of techniques of
    that type
    ```


### <a id="schemas-exceptionspy"></a>[exceptions.py](schemas/exceptions.py)

#### Classes / Классы
- **IncorrectAmountOfWorker**

- **NoAvailableResources**

- **NoDepots**

- **NoSufficientContractorError**

    ```
    Raise when contractor error occurred.
    
    It indicates when the contractors have not sufficient resources to perform schedule.
    ```

- **NotEnoughMaterialsInDepots**


### <a id="schemas-graphpy"></a>[graph.py](schemas/graph.py)

#### Classes / Классы
- **EdgeType**

    ```
    Types of edges in the work graph.
    
    Типы ребер в графе работ.
    ```

- **GraphEdge**

    ```
    Edge connecting two nodes in a work graph.
    
    Ребро, соединяющее две вершины графа работ.
    
    Attributes:
        start (GraphNode): start node.
            Начальная вершина.
        finish (GraphNode): finish node.
            Конечная вершина.
        lag (float | None): delay between nodes.
            Задержка между вершинами.
        type (EdgeType | None): type of connection.
            Тип связи.
    ```

- **GraphNode**

    ```
    Node of a work graph.
    
    Узел графа работ.
    ```

- **WorkGraph**

    ```
    Graph of works with service nodes.
    
    Граф работ с сервисными узлами.
    ```

#### Functions / Функции
- **get_finish_stage**

    ```
    Create a service finish node for the work graph.
    
    Создает сервисный завершающий узел для графа работ.
    
    Args:
        parents (list[GraphNode | tuple[GraphNode, float, EdgeType]]):
            non-service nodes without children.
            несервисные узлы без потомков.
        work_id (str | None): identifier of the finish node.
            Идентификатор завершающего узла.
        rand (Random | None): random generator with fixed seed.
            Генератор случайных чисел с фиксированным зерном.
    
    Returns:
        GraphNode: created finish node.
            Созданный завершающий узел.
    ```

- **get_start_stage**

    ```
    Create a service start node for the work graph.
    
    Создает сервисный стартовый узел для графа работ.
    
    Args:
        work_id (str | None): identifier of the start node.
            Идентификатор стартового узла.
        rand (Random | None): random generator with fixed seed.
            Генератор случайных чисел с фиксированным зерном.
    
    Returns:
        GraphNode: created start node.
            Созданный стартовый узел.
    ```

- **recreate**

- **serialize_wg**


### <a id="schemas-identifiablepy"></a>[identifiable.py](schemas/identifiable.py)

#### Classes / Классы
- **Identifiable**

    ```
    A base class for all unique entities
    
    :param id: unique id for the object
    :param name: name of for the object
    ```


### <a id="schemas-intervalpy"></a>[interval.py](schemas/interval.py)

#### Classes / Классы
- **Interval**

    ```
    Base class for random number generation from distributions.
    
    Базовый класс для генерации случайных чисел по распределениям.
    ```

- **IntervalGaussian**

    ```
    Gaussian distribution interval.
    
    Интервал с нормальным распределением.
    
    Attributes:
        mean (float): distribution mean.
            Среднее распределения.
        sigma (float): distribution variance.
            Дисперсия распределения.
        min_val (float | None): left boundary.
            Левая граница.
        max_val (float | None): right boundary.
            Правая граница.
        rand (Random | None): random generator with seed.
            Генератор случайных чисел с зерном.
    ```

- **IntervalUniform**

    ```
    Uniform distribution interval.
    
    Интервал с равномерным распределением.
    
    Attributes:
        min_val (float): left boundary.
            Левая граница.
        max_val (float): right boundary.
            Правая граница.
        rand (Random | None): random generator with seed.
            Генератор случайных чисел с зерном.
    ```


### <a id="schemas-landscapepy"></a>[landscape.py](schemas/landscape.py)

#### Classes / Классы
- **LandscapeConfiguration**

    ```
    Configuration of resource holders and routes.
    
    Конфигурация держателей ресурсов и маршрутов.
    ```

- **MaterialDelivery**

    ```
    Schedule of material deliveries for a work.
    
    Расписание поставок материалов для работы.
    ```

- **ResourceHolder**

    ```
    Storage node that owns vehicles.
    
    Узел хранения, владеющий транспортом.
    ```

- **ResourceSupply**

    ```
    Base entity that supplies resources.
    
    Базовая сущность, предоставляющая ресурсы.
    ```

- **Road**

    ```
    Road segment between two nodes.
    
    Дорожный сегмент между двумя узлами.
    ```

- **Vehicle**

    ```
    Transport vehicle with material capacity.
    
    Транспортное средство с грузоподъемностью.
    ```


### <a id="schemas-landscape_graphpy"></a>[landscape_graph.py](schemas/landscape_graph.py)

#### Classes / Классы
- **LandEdge**

    ```
    Connection between two vertices of a transport graph.
    
    Соединение между двумя вершинами транспортного графа.
    
    Attributes:
        id (str): identifier of the edge.
            Идентификатор ребра.
        start (LandGraphNode): start node.
            Начальная вершина.
        finish (LandGraphNode): finish node.
            Конечная вершина.
        weight (float): length of the edge.
            Длина ребра.
        bandwidth (int): number of vehicles per hour.
            Количество транспортных средств в час.
    ```

- **LandGraph**

    ```
    Graph representing the landscape transport network.
    
    Граф, представляющий транспортную сеть ландшафта.
    ```

- **LandGraphNode**

    ```
    Participant of the landscape transport network.
    
    Участник транспортной сети ландшафта.
    ```

- **ResourceStorageUnit**

    ```
    Resource storage for a land graph node.
    
    Хранилище ресурсов для узла ландшафтного графа.
    ```


### <a id="schemas-projectpy"></a>[project.py](schemas/project.py)

#### Classes / Классы
- **ScheduledProject**


### <a id="schemas-requirementspy"></a>[requirements.py](schemas/requirements.py)

#### Classes / Классы
- **BaseReq**

    ```
    A class summarizing any requirements for the work to be performed related to renewable and non-renewable
    resources, infrastructure requirements, etc.
    ```

- **ConstructionObjectReq**

    ```
    Requirements for infrastructure and the construction of other facilities: electricity, pipelines, roads, etc
    
    :param kind: type of resource/profession
    :param name: the name of this requirement
    ```

- **EquipmentReq**

    ```
    Requirements for renewable non-human resources: equipment, trucks, machines, etc
    
    :param kind: type of resource/profession
    :param name: the name of this requirement
    ```

- **MaterialReq**

    ```
    Requirements for non-renewable materials: consumables, spare parts, construction materials
    
    :param kind: type of resource/profession
    :param name: the name of this requirement
    ```

- **WorkerReq**

    ```
    Requirements related to renewable human resources
    
    :param kind: type of resource/profession
    :param volume: volume of work in time units
    :param min_count: minimum number of employees needed to perform the work
    :param max_count: maximum allowable number of employees performing the work
    :param name: the name of this requirement
    ```

- **ZoneReq**


### <a id="schemas-resourcespy"></a>[resources.py](schemas/resources.py)

#### Classes / Классы
- **ConstructionObject**

- **EmptySpaceConstructionObject**

- **Equipment**

- **Material**

- **Resource**

    ```
    A class summarizing the different resources used in the work: Human resources, equipment, materials, etc.
    ```

- **Worker**

    ```
    A class dedicated to human resources
    
    :param count: the number of people in this resource
    :param contractor_id: Contractor id if resources are added directly to the contractor
    :param productivity: interval from Gaussian or Uniform distribution, that contains possible values of
    productivity of certain worker
    ```

- **WorkerProductivityMode**


### <a id="schemas-schedulepy"></a>[schedule.py](schemas/schedule.py)

#### Classes / Классы
- **Schedule**

    ```
    Represents work schedule. Is a wrapper around DataFrame with specific structure.
    ```

#### Functions / Функции
- **order_nodes_by_start_time**

    ```
    Makes ScheduledWorks' ordering that satisfies:
    1. Ascending order by start time
    2. Toposort
    
    :param works:
    :param wg:
    :return:
    ```


### <a id="schemas-schedule_specpy"></a>[schedule_spec.py](schemas/schedule_spec.py)

#### Classes / Классы
- **ScheduleSpec**

    ```
    Here is the container for externally given terms, that Schedule should satisfy.
    Must be used in schedulers.
    
    :param work2spec: work specs
    ```

- **WorkSpec**

    ```
    Here are the container for externally given terms, that the resulting `ScheduledWork` should satisfy.
    Must be used in schedulers.
    :param chain: the chain of works, that should be scheduled one after another, e.g. inseparable,
    that starts from this work. Now unsupported.
    :param assigned_workers: predefined worker team (scheduler should assign this worker team to this work)
    :param assigned_time: predefined work time (scheduler should schedule this work with this execution time)
    :param is_independent: should this work be resource-independent, e.g. executing with no parallel users of
    its types of resources
    ```


### <a id="schemas-scheduled_workpy"></a>[scheduled_work.py](schemas/scheduled_work.py)

#### Classes / Классы
- **ScheduledWork**

    ```
    Contains all necessary info to represent WorkUnit in schedule:
    
    * WorkUnit
    * list of workers, that are required to complete task
    * start and end time
    * contractor, that complete task
    * list of equipment, that is needed to complete the task
    * list of materials - set of non-renewable resources
    * object - variable, that is used in landscape
    ```


### <a id="schemas-serializablepy"></a>[serializable.py](schemas/serializable.py)

#### Classes / Классы
- **AutoJSONSerializable**

    ```
    Parent class for serialization of classes, which can be automatically converted to dict with Serializable properties
    and custom (de-)serializers, marked with custom_serializer and custom_deserializer decorators.
    :param JSONSerializable[AJS]:
    :param ABC: helper class to create custom abstract classes
    ```

- **JSONSerializable**

- **Serializable**

    ```
    Parent class for (de-)serialization different data structures.
    
    :param ABC: helper class to create custom abstract classes
    :param Generic[T, S]: base class to make Serializable as universal class, using user's types T, S
    ```

- **StrSerializable**

    ```
    Parent class for serialization of classes, which can be converted to String representation or given from String
    representation
    
    :param Serializable[str, SS]:
    :param ABC: helper class to create custom abstract classes
    :param Generic[SS]: base class to make StrSerializable as universal class,
    using user's types SS and it's descendants
    ```


### <a id="schemas-sorted_listpy"></a>[sorted_list.py](schemas/sorted_list.py)

#### Classes / Классы
- **ExtendedSortedList**


### <a id="schemas-structure_estimatorpy"></a>[structure_estimator.py](schemas/structure_estimator.py)

#### Classes / Классы
- **DefaultStructureEstimator**

- **DefaultStructureGenerationEstimator**

- **StructureEstimator**

- **StructureGenerationEstimator**


### <a id="schemas-timepy"></a>[time.py](schemas/time.py)

#### Classes / Классы
- **Time**

    ```
    Class for describing all basic operations for working with time in framework
    
    :param value: initial time value
    ```


### <a id="schemas-time_estimatorpy"></a>[time_estimator.py](schemas/time_estimator.py)

#### Classes / Классы
- **DefaultWorkEstimator**

- **WorkEstimationMode**

- **WorkTimeEstimator**

    ```
    Implementation of time estimator of work with a given set of resources.
    ```

#### Functions / Функции
- **communication_coefficient**


### <a id="schemas-typespy"></a>[types.py](schemas/types.py)

#### Classes / Классы
- **EventType**

- **ScheduleEvent**


### <a id="schemas-utilspy"></a>[utils.py](schemas/utils.py)

#### Functions / Функции
- **uuid_str**

    ```
    Transform str to uuid format.
    ```


### <a id="schemas-workspy"></a>[works.py](schemas/works.py)

#### Classes / Классы
- **WorkUnit**

    ```
    Class that describe vertex in graph (one work/task)
    ```


### <a id="schemas-zonespy"></a>[zones.py](schemas/zones.py)

#### Classes / Классы
- **DefaultZoneStatuses**

    ```
    Statuses: 0 - not stated, 1 - opened, 2 - closed
    ```

- **Zone**

- **ZoneConfiguration**

- **ZoneStatuses**

- **ZoneTransition**


## <a id="structurator"></a>structurator

### <a id="structurator-basepy"></a>[base.py](structurator/base.py)

#### Functions / Функции
- **fill_parents_to_new_nodes**

    ```
    Restores parent edges for a node split into stages.
    
    Восстанавливает связи с родительскими узлами для узла, разделённого на стадии.
    
    Args:
        origin_node (GraphNode): The original unconverted node.
            Исходный узел до преобразования.
        id2new_nodes (dict[str, GraphNode]): Mapping of new node IDs to nodes.
            Сопоставление новых идентификаторов узлам.
        restructuring_edges2new_nodes_id (dict[tuple[str, str], str]):
            Mapping between original edges and IDs of new nodes that replace them.
            Сопоставление исходных рёбер и идентификаторов новых узлов, заменяющих их.
        use_lag_edge_optimization (bool): Whether to account for lags in edges.
            Учитывать ли задержки в рёбрах.
    
    Returns:
        None: This function modifies ``id2new_nodes`` in-place.
            None: функция изменяет ``id2new_nodes`` на месте.
    ```

- **graph_restructuring**

    ```
    Converts a work graph to use only FS and IFS edges.
    
    Преобразует рабочий граф, оставляя только связи Finish-Start и
    Inseparable-Finish-Start.
    
    Args:
        wg (WorkGraph): The graph to convert.
            Преобразуемый граф.
        use_lag_edge_optimization (bool, optional): Whether to account for lag
            values on edges. Defaults to ``False``.
            Учитывать ли задержки на рёбрах. По умолчанию ``False``.
    
    Returns:
        WorkGraph: Restructured work graph.
            WorkGraph: реструктурированный граф.
    ```

- **make_new_node_id**

    ```
    Creates an auxiliary ID for restructuring the graph.
    
    Создаёт вспомогательный идентификатор для реструктуризации графа.
    
    Args:
        work_unit_id (str): ID of the work unit.
            Идентификатор работы.
        ind (int): Sequence number of the work unit stage.
            Порядковый номер этапа работы.
    
    Returns:
        str: Auxiliary ID for the work unit.
            Вспомогательный идентификатор работы.
    ```

- **split_node_into_stages**

    ```
    Splits a work node into sequential stages.
    
    Разделяет узел работы на последовательные стадии.
    
    The function creates intermediate nodes according to restructuring edges and
    connects them with ``InseparableFinishStart`` edges. The last stage keeps the
    original node ID to simplify parent restoration.
    
    Функция создаёт промежуточные узлы в соответствии с рёбрами реструктуризации
    и соединяет их рёбрами ``InseparableFinishStart``. Последняя стадия сохраняет
    исходный идентификатор узла для упрощения восстановления родителей.
    
    Args:
        origin_node (GraphNode): Node to be divided into stages.
            Узел, который требуется разделить на стадии.
        restructuring_edges (list[tuple[GraphEdge, bool]]):
            Restructuring edges with a flag showing direction.
            Рёбра реструктуризации и флаг направления.
        id2new_nodes (dict[str, GraphNode]): Mapping for storing created nodes.
            Отображение для хранения созданных узлов.
        restructuring_edges2new_nodes_id (dict[tuple[str, str], str]):
            Mapping from original edges to new node IDs.
            Сопоставление исходных рёбер с идентификаторами новых узлов.
        use_lag_edge_optimization (bool): Whether to handle lag edges explicitly.
            Учитывать ли задержки в рёбрах явно.
    
    Returns:
        None: Function modifies mappings in-place.
            None: функция изменяет отображения на месте.
    ```


### <a id="structurator-delete_graph_nodepy"></a>[delete_graph_node.py](structurator/delete_graph_node.py)

#### Functions / Функции
- **_node_deletion**

- **delete_graph_node**

    ```
    Deletes a task from WorkGraph.
    If the task consists of several inseparable nodes this function deletes all of those nodes
    :param original_wg: WorkGraph from which a task is deleted
    :param remove_gn_id: id of the node, corresponding to the deleted task.
    If the task consists of several inseparable nodes, this is id of one of them
    :param change_id: do ids in the new graph need to be changed
    :return: new WorkGraph with deleted task
    ```


### <a id="structurator-graph_insertionpy"></a>[graph_insertion.py](structurator/graph_insertion.py)

#### Functions / Функции
- **graph_in_graph_insertion**

    ```
    Inserts the slave WorkGraph into the master WorkGraph,
    while the starting vertex slave_wg becomes the specified master_start,
    and the finishing vertex is correspondingly master_finish
    :param master_wg: the WorkGraph into which the insertion is performed
    :param master_start: GraphNode which will become the parent for the entire slave_wg
    :param master_finish: GraphNode which will become a child for the whole slave_wg
    :param slave_wg: WorkGraph to be inserted into master_wg
    :param change_id: do ids in the new graph need to be changed
    :return: new union WorkGraph
    ```


### <a id="structurator-insert_wupy"></a>[insert_wu.py](structurator/insert_wu.py)

#### Functions / Функции
- **_new_edges**

- **_reduce_to_tuple_type**

- **insert_work_unit**

    ```
    Inserts new node in the WorkGraph, based on given WorkUnit
    :param original_wg: WorkGraph into which we insert new node
    :param inserted_wu: WorkUnit on the basis of which we create new GraphNode
    :param parents_edges: nodes which are supposed to be the parents of new GraphNode
    :param children_edges: nodes which are supposed to be the children of new GraphNode
    :param change_id: do ids in the new graph need to be changed
    :return: new WorkGraph with inserted new node
    ```


### <a id="structurator-light_modificationspy"></a>[light_modifications.py](structurator/light_modifications.py)

#### Functions / Функции
- **work_graph_ids_simplification**

    ```
    Creates a new WorkGraph with simplified numeric ids (numeric ids are converted to a string)
    :param wg: original WorkGraph
    :param id_offset: start for numbering new ids
    :param change_id: Do IDs in the new graph need to be changed
    :return: new WorkGraph with numeric ids
    ```


### <a id="structurator-prepare_wg_copypy"></a>[prepare_wg_copy.py](structurator/prepare_wg_copy.py)

#### Functions / Функции
- **copy_graph_node**

    ```
    Makes a deep copy of GraphNode without edges. It's id can be changed to a new randomly generated or specified one
    :param node: original GraphNode
    :param new_id: specified new id
    :param change_id: do ids in the new graph need to be changed
    :return: copy of GraphNode and pair(old node id, new node id)
    ```

- **new_start_finish**

    ```
    Prepares new start and finish to create WorkGraph after copying it
    :param original_wg: WorkGraph, on which base prepare_work_graph_copy was run
    :param copied_nodes: New nodes, on which to create new WorkGraph
    :param old_to_new_ids: Dictionary to translate old nodes to new, using their IDs
    :return: new start and new finish nodes, on the base of which to create a WorkGraph
    ```

- **prepare_work_graph_copy**

    ```
    Makes a deep copy of the GraphNodes of the original graph with new ids and updated edges,
    ignores all GraphNodes specified in the exception list and GraphEdges associated with them
    :param wg: original WorkGraph for copy
    :param excluded_nodes: GraphNodes to be excluded from the graph
    :param use_ids_simplification: If true, creates short numeric ids converted to strings,
    otherwise uses uuid to generate id
    :param id_offset: Shift for numeric ids, used only if param use_ids_simplification is True
    :param change_id: Do IDs in the new graph need to be changed
    :return: A dictionary with GraphNodes by their id
    and a dictionary linking the ids of GraphNodes of the original graph and the new GraphNode ids
    ```

- **restore_parents**

    ```
    Restores edges in GraphNode for copied WorkGraph with changed ids
    :param new_nodes: needed copied nodes
    :param original_wg: original WorkGraph for edge restoring for new nodes
    :param excluded_ids: dictionary of relationships between old ids and new ids
    :param old_to_new_ids: a dictionary linking the ids of GraphNodes of the original graph and the new GraphNode ids
    :return:
    ```


## <a id="userinputparser"></a>userinput/parser

### <a id="userinputparser-contractor_typepy"></a>[contractor_type.py](userinput/parser/contractor_type.py)

#### Classes / Классы
- **ContractorType**

    ```
    Levels of contractor performance.
    
    Уровни производительности подрядчика.
    ```


### <a id="userinputparser-csv_parserpy"></a>[csv_parser.py](userinput/parser/csv_parser.py)

#### Classes / Классы
- **CSVParser**

    ```
    Parser for reading work graphs and contractor data from CSV.
    
    Парсер для чтения графов работ и данных подрядчиков из CSV.
    ```


### <a id="userinputparser-exceptionpy"></a>[exception.py](userinput/parser/exception.py)

#### Classes / Классы
- **InputDataException**

    ```
    Raised when information about task links is missing.
    
    Возникает при отсутствии информации о связях задач.
    ```

- **WorkGraphBuildingException**

    ```
    Raised when work graph can't be built.
    
    Возникает, когда невозможно построить граф работ.
    ```


### <a id="userinputparser-general_buildpy"></a>[general_build.py](userinput/parser/general_build.py)

#### Classes / Классы
- **Graph**

    ```
    Simple directed graph for detecting and removing cycles.
    
    Простой ориентированный граф для обнаружения и удаления циклов.
    ```

#### Functions / Функции
- **add_graph_info**

    ```
    Filter nonexistent predecessors and collect edge info.
    
    Отфильтровывает несуществующих предшественников и собирает информацию о рёбрах.
    
    Args:
        frame (pd.DataFrame): Preprocessed DataFrame.
            Предварительно обработанный DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame enriched with edge tuples.
            pd.DataFrame: DataFrame, дополненный кортежами рёбер.
    ```

- **break_loops_in_input_graph**

    ```
    Remove cycles from the input work graph.
    
    Удаляет циклы из входного графа работ.
    
    The algorithm deletes the edge with the smallest weight within each
    detected cycle (e.g., link frequency in history).
    
    Алгоритм удаляет ребро с наименьшим весом в каждом найденном цикле
    (например, частота связи в истории).
    
    Args:
        works_info (pd.DataFrame): Input work information.
            Входные данные о работах.
    
    Returns:
        pd.DataFrame: Work info without cycles.
            pd.DataFrame: данные о работах без циклов.
    ```

- **build_work_graph**

    ```
    Construct a work graph from DataFrame data.
    
    Создаёт граф работ из данных DataFrame.
    
    Args:
        frame (pd.DataFrame): DataFrame with works and edges.
            DataFrame с работами и рёбрами.
        resource_names (list[str]): Names of resources.
            Названия ресурсов.
        work_estimator (WorkTimeEstimator): Estimator of work resources.
            Оценщик ресурсов для работ.
    
    Returns:
        WorkGraph: Built work graph.
            WorkGraph: построенный граф.
    ```

- **fix_df_column_with_arrays**

    ```
    Convert comma-separated strings in a column to lists.
    
    Преобразует строки с разделителями в списки.
    
    Args:
        column (pd.Series): Column with comma-separated values.
            Колонка со значениями, разделёнными запятыми.
        cast (Callable[[str], Any] | None): Function for element conversion.
            Функция преобразования элемента.
        none_elem (Any | None): Placeholder for missing elements.
            Значение-заглушка для отсутствующих элементов.
    
    Returns:
        pd.Series: Converted column.
            pd.Series: преобразованная колонка.
    ```

- **get_graph_contractors**

    ```
    Read contractor information from a CSV file.
    
    Считывает информацию о подрядчике из CSV-файла.
    
    Args:
        path (str): Path to the CSV with worker counts.
            Путь к CSV с количеством рабочих.
        contractor_name (str | None): Name of the contractor.
            Имя подрядчика.
    
    Returns:
        tuple[list[Contractor], dict[str, float]]: Contractors and workers capacity.
            tuple[list[Contractor], dict[str, float]]: подрядчики и их мощность по рабочим.
    ```

- **preprocess_graph_df**

    ```
    Prepare work graph data for building.
    
    Подготавливает данные графа работ для построения.
    
    Args:
        frame (pd.DataFrame): Raw work information.
            Исходные данные о работах.
        name_mapper (NameMapper | None): Mapper of activity names.
            Сопоставитель названий работ.
    
    Returns:
        pd.DataFrame: Normalized DataFrame ready for processing.
            pd.DataFrame: нормализованный DataFrame для обработки.
    ```

- **topsort_graph_df**

    ```
    Sort works in topological order.
    
    Сортирует работы в топологическом порядке.
    
    Args:
        frame (pd.DataFrame): DataFrame of works.
            DataFrame работ.
    
    Returns:
        pd.DataFrame: Topologically sorted DataFrame.
            pd.DataFrame: топологически отсортированный DataFrame.
    ```


### <a id="userinputparser-historypy"></a>[history.py](userinput/parser/history.py)

#### Functions / Функции
- **find_min_without_outliers**

    ```
    Find the minimal value excluding outliers.
    
    Находит минимальное значение без учёта выбросов.
    
    Args:
        lst (list[float]): Input values.
            Входные значения.
    
    Returns:
        float: Minimal value.
            float: минимальное значение.
    ```

- **gather_links_types_statistics**

    ```
    Count statistics of mutual task arrangements.
    
    Подсчитывает статистику взаимного расположения задач.
    
    Args:
        s1 (str): Start of first work.
            Начало первой работы.
        f1 (str): Finish of first work.
            Завершение первой работы.
        s2 (str): Start of second work.
            Начало второй работы.
        f2 (str): Finish of second work.
            Завершение второй работы.
    
    Returns:
        Tuple[int, int, int, list, list, int, list, list, int, list, list, int, list, list]:
            Statistics of mutual arrangements.
            Tuple[...] : статистика взаимных расположений.
    ```

- **get_all_connections**

    ```
    Generate all unique pairs of works.
    
    Формирует все уникальные пары работ.
    
    Args:
        graph_df (pd.DataFrame): DataFrame with work graph.
            DataFrame с графом работ.
        use_mapper (bool): Whether to translate task names.
            Преобразовывать ли имена задач.
        mapper (NameMapper | None): Mapper for translating names.
            Сопоставитель имён.
    
    Returns:
        Tuple[dict[str, list], dict[str, list]]: IDs and names of work pairs.
            Tuple[dict[str, list], dict[str, list]]: идентификаторы и имена пар работ.
    ```

- **get_all_seq_statistic**

    ```
    Compute connection statistics between tasks.
    
    Вычисляет статистику связей между задачами.
    
    Args:
        history_data (pd.DataFrame): Historical schedule data.
            Исторические данные расписания.
        graph_df (pd.DataFrame): Work graph data.
            Данные графа работ.
        use_model_name (bool): Use model names instead of granular names.
            Использовать ли модельные имена вместо подробных.
        mapper (NameMapper | None): Name mapper.
            Сопоставитель имён.
    
    Returns:
        dict[str, list]: Mapping from task ID to connection info.
            dict[str, list]: отображение от ID задачи к информации о связях.
    ```

- **get_delta_between_dates**

    ```
    Calculate days between two dates in ``YYYY-MM-DD`` format.
    
    Вычисляет количество дней между двумя датами в формате ``ГГГГ-ММ-ДД``.
    
    Args:
        first (str): First date.
            Первая дата.
        second (str): Second date.
            Вторая дата.
    
    Returns:
        int: Number of days, at least 1.
            int: количество дней, минимум 1.
    ```

- **set_connections_info**

    ```
    Restore task connections using historical data.
    
    Восстанавливает связи задач с помощью исторических данных.
    
    Args:
        graph_df (pd.DataFrame): Work graph info.
            Информация о графе работ.
        history_data (pd.DataFrame): Historical connection data.
            Исторические данные о связях.
        use_model_name (bool): Use model names in history.
            Использовать ли модельные имена в истории.
        mapper (NameMapper | None): Name mapper.
            Сопоставитель имён.
        change_connections_info (bool): Modify existing connection info.
            Изменять ли существующую информацию о связях.
        all_connections (bool): Replace all existing connections.
            Заменять ли все существующие связи.
        id2ind (dict[str, int] | None): Mapping from task ID to index.
            Отображение идентификатора задачи в индекс.
    
    Returns:
        pd.DataFrame: DataFrame with restored connections.
            pd.DataFrame: DataFrame с восстановленными связями.
    ```


## <a id="utilities"></a>utilities

### <a id="utilities-base_optpy"></a>[base_opt.py](utilities/base_opt.py)

#### Functions / Функции
- **coordinate_descent**

- **dichotomy_float**

- **dichotomy_int**


### <a id="utilities-collections_utilpy"></a>[collections_util.py](utilities/collections_util.py)

#### Functions / Функции
- **build_index**

    ```
    :param items: an iterable to index
    :param key_getter: a function that should retrieve index key from item
    :param value_getter: a function that should retrieve index value from item
    :return: dictionary that represents built index given by `key_getter` function
    ```

- **flatten**

    ```
    Returns a generator which should flatten any heterogeneous iterable
    
    :param xs:
    :return:
    ```

- **reverse_dictionary**


### <a id="utilities-datetime_utilpy"></a>[datetime_util.py](utilities/datetime_util.py)

#### Functions / Функции
- **add_time_delta**

    ```
    Adds time delta to base datetime
    
    :param base_datetime:
    :param time_delta:
    :param time_units: can be days, seconds, microseconds, milliseconds, minutes, hours, weeks
    :return:
    ```

- **parse_datetime**

    ```
    Parses datetime from string
    
    :param dts: String datetime
    :param date_format: String format. If not provided, '%Y-%m-%d' and then '%y-%m-%d %H:%M:%S' are tried.
    :return:
    ```


### <a id="utilities-linked_listpy"></a>[linked_list.py](utilities/linked_list.py)

#### Classes / Классы
- **Iterator**

- **LinkedList**

- **Node**


### <a id="utilities-name_mapperpy"></a>[name_mapper.py](utilities/name_mapper.py)

#### Classes / Классы
- **DictNameMapper**

- **DummyNameMapper**

- **ModelNameMapper**

    ```
    NameMapper for Kovalchuk's model integration
    ```

- **NameMapper**

#### Functions / Функции
- **get_inverse_task_name_mapping**

    ```
    Gets mapping of the unique names to our task names
    :param path: path to the csv file
    :return: dict {unique_name: our_name}
    ```

- **get_task_name_unique_mapping**

    ```
    Gets mapping of our task names to the unique names
    :param path: path to the csv file
    :return: dict {our_name: unique_name}
    ```

- **read_json**

    ```
    Gets mapping of the unique names to our task names
    :param path: path to the .json file
    :return: NameMapper: our_name -> unique_name
    ```

- **read_tasks_df**

    ```
    Reads DataFrame with tasks
    :param path: path to the csv file
    :return: The DataFrame read
    ```


### <a id="utilities-prioritypy"></a>[priority.py](utilities/priority.py)

#### Functions / Функции
- **check_and_correct_priorities**

- **extract_priority_groups_from_indices**

- **extract_priority_groups_from_nodes**

- **update_priority**


### <a id="utilities-priority_queuepy"></a>[priority_queue.py](utilities/priority_queue.py)

#### Classes / Классы
- **PriorityQueue**


### <a id="utilities-resource_usagepy"></a>[resource_usage.py](utilities/resource_usage.py)

#### Functions / Функции
- **get_resources_peak_usage**

- **get_total_resources_usage**

- **resources_costs_sum**

    ```
    Count the summary cost of resources in received schedule
    ```

- **resources_peaks_sum**

    ```
    Count the summary of resources peaks usage in received schedule
    ```

- **resources_sum**

    ```
    Count the summary usage of resources in received schedule
    ```


### <a id="utilities-schedulepy"></a>[schedule.py](utilities/schedule.py)

#### Functions / Функции
- **fix_split_tasks**

    ```
    Process and merge information for all tasks, which were separated on the several stages during split
    
    :param baps_schedule_df: pd.DataFrame: schedule with info for tasks separated on stages
    :return: pd.DataFrame: schedule with merged info for all real tasks
    ```

- **merge_split_stages**

    ```
    Merge split stages of the same real task into one
    
    :param task_df: pd.DataFrame: one real task's stages dataframe, sorted by start time
    :return: pd.Series with the full information about the task
    ```

- **offset_schedule**

    ```
    Returns full schedule object with `start` and `finish` columns pushed by date in `offset` argument.
    :param schedule: the schedule itself
    :param offset: Start of schedule, to add as an offset.
    :return: Shifted schedule DataFrame.
    ```


### <a id="utilities-serializerspy"></a>[serializers.py](utilities/serializers.py)

#### Functions / Функции
- **_decorate_serializer**

- **custom_field_deserializer**

- **custom_field_serializer**

- **custom_serializer**

    ```
    Meta-decorator for marking custom serializers or deserializers methods.<br/>
    This decorator can stack with other serializer/deserializer decorators.
    :param type_or_field: Name (str) of field or type (type) of fields, which will be serialized with this serializer in
    current class. If type should be presented in str representation, consider using custom_type_serializer or
    custom_type_deserializer decorators.
    :param deserializer:
    If True, the decorated function will be considered as a custom deserializer for type_or_field type or field<br/>
    If None, deserializer should be decorated separately with custom_serializer or custom_field_deserializer or
    custom_type_deserializer decorator
    :return:
    ```

- **custom_type_deserializer**

- **custom_type_serializer**

- **default_dataframe_deserializer**

- **default_dataframe_serializer**

- **default_ndarray_deserializer**

- **default_ndarray_serializer**

- **default_np_int_deserializer**

- **default_np_int_serializer**

- **default_np_long_deserializer**

- **default_np_long_serializer**


### <a id="utilities-validationpy"></a>[validation.py](utilities/validation.py)

#### Functions / Функции
- **_check_all_allocated_workers_do_not_exceed_capacity_of_contractors**

- **_check_all_tasks_have_valid_duration**

- **_check_all_tasks_scheduled**

- **_check_all_workers_correspond_to_worker_reqs**

- **_check_parent_dependencies**

- **check_all_allocated_workers_do_not_exceed_capacity_of_contractors**

- **validate_schedule**

    ```
    Checks if the schedule is correct and can be executed.
    If there is an error, this function raises AssertException with an appropriate message
    If it finishes without any exception, it means successful passing of the verification
    
    :param contractors:
    :param wg:
    :param schedule: to apply verification
    ```


## <a id="utilitiessampler"></a>utilities/sampler

### <a id="utilitiessampler-__init__py"></a>[__init__.py](utilities/sampler/__init__.py)

#### Classes / Классы
- **Sampler**

    ```
    Generates random work units and graph nodes.
    
    Generates random work units and graph nodes.
    Генерирует случайные рабочие единицы и узлы графа.
    ```


### <a id="utilitiessampler-requirementspy"></a>[requirements.py](utilities/sampler/requirements.py)

#### Functions / Функции
- **get_worker_req**

    ```
    Generate requirement for a single worker type.
    
    Generate requirement for a single worker type.
    Сгенерировать требование для одного типа рабочего.
    
    Args:
        rand: Random generator. rand: Генератор случайных чисел.
        name: Worker specialization. name: Специализация рабочего.
        volume: Range of required volume. volume: Диапазон требуемого объема.
        worker_count: Range of workers per unit. worker_count: Диапазон
            рабочих на единицу.
    
    Returns:
        WorkerReq: Requirement description. WorkerReq: Описание
            требования.
    ```

- **get_worker_reqs_list**

    ```
    Generate list of random worker requirements.
    
    Generate list of random worker requirements.
    Сгенерировать список случайных требований к рабочим.
    
    Args:
        rand: Random generator. rand: Генератор случайных чисел.
        volume: Range of required volume. volume: Диапазон требуемого
            объема.
        worker_count: Range of workers per unit. worker_count: Диапазон
            рабочих на единицу.
    
    Returns:
        list[WorkerReq]: Worker requirements list. list[WorkerReq]: Список
            требований к рабочим.
    ```

- **get_worker_specific_reqs_list**

    ```
    Generate requirements for specific worker types.
    
    Generate requirements for specific worker types.
    Сгенерировать требования для конкретных типов рабочих.
    
    Args:
        rand: Random generator. rand: Генератор случайных чисел.
        worker_names: Specializations list. worker_names: Список специализаций.
        volume: Range of required volume. volume: Диапазон требуемого
            объема.
        worker_count: Range of workers per unit. worker_count: Диапазон
            рабочих на единицу.
    
    Returns:
        list[WorkerReq]: Worker requirements list. list[WorkerReq]: Список
            требований к рабочим.
    ```


### <a id="utilitiessampler-typespy"></a>[types.py](utilities/sampler/types.py)

#### Classes / Классы
- **MinMax**

    ```
    Range with minimum and maximum values.
    
    Range with minimum and maximum values.
    Диапазон с минимальным и максимальным значениями.
    
    Attributes:
        min: Lower bound. min: Нижняя граница.
        max: Upper bound. max: Верхняя граница.
    ```


### <a id="utilitiessampler-workspy"></a>[works.py](utilities/sampler/works.py)

#### Functions / Функции
- **get_similar_work_unit**

    ```
    Generate work unit similar to exemplar.
    
    Generate work unit similar to exemplar.
    Сгенерировать рабочую единицу, подобную образцу.
    
    Args:
        rand: Random generator. rand: Генератор случайных чисел.
        exemplar: Base work unit. exemplar: Базовая рабочая единица.
        scalar: Scale factor for volume. scalar: Коэффициент масштабирования
            объема.
        name: New name if provided. name: Новое имя, если указано.
        work_id: New identifier if provided. work_id: Новый идентификатор,
            если указан.
    
    Returns:
        WorkUnit: Generated work unit. WorkUnit: Сгенерированная рабочая
            единица.
    ```

- **get_work_unit**

    ```
    Generate a random work unit.
    
    Generate a random work unit.
    Сгенерировать случайную рабочую единицу.
    
    Args:
        rand: Random generator. rand: Генератор случайных чисел.
        name: Name of work unit. name: Название рабочей единицы.
        work_id: Identifier of work unit. work_id: Идентификатор
            рабочей единицы.
        volume_type: Unit of volume. volume_type: Единица измерения объема.
        group: Group of work. group: Группа работы.
        work_volume: Range of work volume. work_volume: Диапазон объема
            работ.
        req_volume: Range of requirement volume. req_volume: Диапазон
            объема требований.
        req_worker_count: Range of worker numbers per requirement.
            req_worker_count: Диапазон чисел рабочих на требование.
    
    Returns:
        WorkUnit: Generated work unit. WorkUnit: Сгенерированная рабочая
            единица.
    ```


## <a id="utilitiesvisualization"></a>utilities/visualization

### <a id="utilitiesvisualization-__init__py"></a>[__init__.py](utilities/visualization/__init__.py)

#### Classes / Классы
- **ScheduleVisualization**

- **Visualization**

- **WorkGraphVisualization**


### <a id="utilitiesvisualization-basepy"></a>[base.py](utilities/visualization/base.py)

#### Classes / Классы
- **VisualizationMode**

#### Functions / Функции
- **visualize**

    ```
    Visualizes the figure according to the provided settings
    :param fig: The figure
    :param mode: Visualisation mode. Can be bitwise-or (|) of several modes.
    :param file_name: Optional name of a saved file. Passed, if SaveFig in mode.
    :return: The figure, if ReturnFig in mode. Otherwise, None.
    ```


### <a id="utilitiesvisualization-resourcespy"></a>[resources.py](utilities/visualization/resources.py)

#### Classes / Классы
- **EmploymentFigType**

#### Functions / Функции
- **convert_schedule_df**

- **create_employment_fig**

- **get_resources**

- **get_schedule_df**

- **get_workers_intervals**

- **resource_employment_fig**


### <a id="utilitiesvisualization-schedulepy"></a>[schedule.py](utilities/visualization/schedule.py)

#### Functions / Функции
- **schedule_gant_chart_fig**

    ```
    Creates and saves a gant chart of the scheduled tasks to the specified path.
    
    :param fig_file_name:
    :param visualization:
    :param remove_service_tasks:
    :param schedule_dataframe: Pandas DataFrame with the information about schedule
    :param color_type defines what tasks color means
    ```


### <a id="utilitiesvisualization-work_graphpy"></a>[work_graph.py](utilities/visualization/work_graph.py)

#### Functions / Функции
- **ax_add_dependencies**

- **ax_add_works**

- **calculate_work_volume**

- **collect_jobs**

- **color_from_str**

- **default_job2text**

- **draw_arrow_between_jobs**

- **empty_job2text**

- **extract_cluster_name**

- **middle_color**

- **setup_jobs**

- **work_graph_fig**

