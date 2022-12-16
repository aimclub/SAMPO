from sampo.schemas.requirements import WorkerReq
from sampo.schemas.time import Time


def scale_reqs(req_list: list[WorkerReq], scalar: float, new_name: str | None = None) -> list[WorkerReq]:
    """
    Multiplies each of the requirements of their list, scaling the maximum number of resources and volume
    :param req_list: list of resource requirements
    :param scalar: scalar to which the requirements are applied
    :param new_name: A new name for the requirements
    :return: scaled requirements
    """
    return [work_req.scale_all(scalar, new_name) for work_req in req_list]


def mul_volume_reqs(req_list: list[WorkerReq], scalar: float, new_name: str | None = None) -> list[WorkerReq]:
    """
    Multiplies each of the requirements of their list, scaling only the volume
    :param req_list: list of resource requirements
    :param scalar: scalar to which the requirements are applied
    :param new_name: A new name for the requirements
    :return: scaled requirements
        """
    return [work_req.scale_volume(scalar, new_name) for work_req in req_list]


def get_borehole_volume(borehole_count: int, base: (float, float)) -> float:
    """
    Function to calculate the scalar for objects depending on the number of boreholes
    :param borehole_count: number of boreholes on the site
    :param base:
        base[0] part of the volume independent of the number of boreholes,
        base[1] part of the volume dependent on the number of boreholes
    :return: returns a scalar to calculate the requirements
    """
    return base[0] + base[1] * borehole_count


def mul_borehole_volume(req_list: list[WorkerReq], borehole_count: int, base: (float, float)) -> list[WorkerReq]:
    """
    Function for scaling resource requirements for works dependent on the number of boreholes
    :param req_list: list of resource requirements
    :param borehole_count: number of boreholes on the site
    :param base:
        base[0] part of the volume independent of the number of boreholes,
        base[1] part of the volume dependent on the number of boreholes
    :return: scaled requirements
    """
    return mul_volume_reqs(req_list, get_borehole_volume(borehole_count, base))


START_PROJECT = []
END_PROJECT = []


# <--road_block-->
# scaled by: scale_reqs
# measurement base: 1 km
MIN_ROAD = [
    WorkerReq(kind="driver", volume=Time(15), min_count=3,
              max_count=9, name="min_road"),
    WorkerReq(kind="manager", volume=Time(5), min_count=1,
              max_count=3, name="min_road"),
    WorkerReq(kind="handyman", volume=Time(30), min_count=6,
              max_count=18, name="min_road"),
]
ATOMIC_ROAD_LEN = 0.05
TEMP_ROAD = mul_volume_reqs(MIN_ROAD, scalar=4, new_name="temp_road")
MIN_TEMP_ROAD = mul_volume_reqs(MIN_ROAD, scalar=5, new_name="temp_road")
FINAL_ROAD = mul_volume_reqs(MIN_ROAD, scalar=10, new_name="final_road")
# /<--road_block-->


# <--engineering_preparation-->
# scaled by: mul_borehole_volume
# measurement base: tuple(volume_without_boreholes, volume_for_one_borehole)
ENGINEERING_PREPARATION_BASE = (0.44, 0.03)
ENGINEERING_PREPARATION = [
    WorkerReq(kind="driver", volume=Time(900), min_count=6,
              max_count=30, name="engineering_preparation"),
    WorkerReq(kind="manager", volume=Time(150), min_count=2,
              max_count=6, name="engineering_preparation"),
    WorkerReq(kind="handyman", volume=Time(900), min_count=6,
              max_count=30, name="engineering_preparation"),
    WorkerReq(kind="engineer", volume=Time(150), min_count=2,
              max_count=6, name="engineering_preparation"),
]
# /<--engineering_preparation-->


# <--power_line-->
# scaled by: scale_reqs
# measurement base: 1 km
POWER_LINE = [
    WorkerReq(kind="driver", volume=Time(120), min_count=6,
              max_count=9, name="power_line"),
    WorkerReq(kind="fitter", volume=Time(120), min_count=6,
              max_count=9, name="power_line"),
    WorkerReq(kind="manager", volume=Time(40), min_count=2,
              max_count=6, name="power_line"),
    WorkerReq(kind="handyman", volume=Time(120), min_count=6,
              max_count=9, name="power_line"),
    WorkerReq(kind="electrician", volume=Time(40), min_count=2,
              max_count=6, name="power_line"),
]
HIGH_POWER_LINE = mul_volume_reqs(POWER_LINE, scalar=1.5, new_name="high_power_line")
# /<--power_line-->

# <--pipe_line-->
# scaled by: scale_reqs
# measurement base: 1 km
PIPE_LINE = scale_reqs(POWER_LINE, scalar=1, new_name="pipe_line")
LOOPING = mul_volume_reqs(PIPE_LINE, scalar=0.8, new_name="looping")
# /<--pipe_line-->
# <--pipe_node-->
# scaled by: -
# measurement base: -
PIPE_NODE = [
    WorkerReq(kind="driver", volume=Time(400), min_count=2,
              max_count=6, name="pipe_node"),
    WorkerReq(kind="fitter", volume=Time(800), min_count=4,
              max_count=8, name="pipe_node"),
    WorkerReq(kind="manager", volume=Time(400), min_count=2,
              max_count=8, name="pipe_node"),
    WorkerReq(kind="handyman", volume=Time(1200), min_count=6,
              max_count=12, name="pipe_node"),
    WorkerReq(kind="electrician", volume=Time(300), min_count=2,
              max_count=6, name="pipe_node"),
]
# /<--pipe_node-->


# <--metering_install-->
# scaled by: -
# measurement base: -
METERING_INSTALL = [
    WorkerReq(kind="driver", volume=Time(200), min_count=2,
              max_count=6, name="metering_install"),
    WorkerReq(kind="fitter", volume=Time(300), min_count=3,
              max_count=9, name="metering_install"),
    WorkerReq(kind="manager", volume=Time(50), min_count=1,
              max_count=2, name="metering_install"),
    WorkerReq(kind="handyman", volume=Time(300), min_count=4,
              max_count=12, name="metering_install"),
    WorkerReq(kind="engineer", volume=Time(100), min_count=2,
              max_count=4, name="metering_install"),
]
# /<--metering_install-->
# <--ktp_nep-->
# scaled by: mul_borehole_volume
# measurement base: tuple(volume_without_boreholes, volume_for_one_borehole)
KTP_NEP_BASE = (0.58, 0.03)
KTP_NEP = [
    WorkerReq(kind="driver", volume=Time(200), min_count=2,
              max_count=6, name="ktp_nep"),
    WorkerReq(kind="fitter", volume=Time(300), min_count=3,
              max_count=9, name="ktp_nep"),
    WorkerReq(kind="manager", volume=Time(50), min_count=1,
              max_count=2, name="ktp_nep"),
    WorkerReq(kind="handyman", volume=Time(300), min_count=4,
              max_count=12, name="ktp_nep"),
    WorkerReq(kind="electrician", volume=Time(100), min_count=2,
              max_count=4, name="ktp_nep"),
]
# <--ktp_nep-->
# scaled by: mul_volume_reqs
# measurement base: DRAINAGE_TANK_BASE_VOLUME + DRAINAGE_TANK_INPUT_VOLUME * borehole_count
# scaled by: mul_borehole_volume
# measurement base: tuple(volume_without_boreholes, volume_for_one_borehole)
DRAINAGE_TANK_BASE = (0.3, 0.05)
DRAINAGE_TANK = [
    WorkerReq(kind="driver", volume=Time(400), min_count=4,
              max_count=8, name="drainage_tank"),
    WorkerReq(kind="manager", volume=Time(50), min_count=1,
              max_count=2, name="drainage_tank"),
    WorkerReq(kind="handyman", volume=Time(300), min_count=3,
              max_count=9, name="drainage_tank"),
]

# <--boreholes_equipment_shared-->
# scaled by: -
# measurement base: -
WATER_BLOCK = scale_reqs(METERING_INSTALL, scalar=1.1, new_name="water_block")
AUTOMATION_BLOCK = scale_reqs(METERING_INSTALL, scalar=0.9, new_name="automation_block")
BLOCK_DOSAGE = scale_reqs(METERING_INSTALL, scalar=1, new_name="block_dosage")
START_FILTER = scale_reqs(METERING_INSTALL, scalar=1.05, new_name="start_filter_system")
FIREWALL = scale_reqs(DRAINAGE_TANK, scalar=1.05, new_name="firewall_tank")
# /<--boreholes_equipment_shared-->


# <--borehole-->
# scaled by: -
# measurement base: -
BOREHOLE = [
    WorkerReq(kind="driver", volume=Time(200), min_count=4,
              max_count=12, name="borehole"),
    WorkerReq(kind="fitter", volume=Time(150), min_count=3,
              max_count=9, name="borehole"),
    WorkerReq(kind="manager", volume=Time(25), min_count=1,
              max_count=3, name="borehole"),
    WorkerReq(kind="handyman", volume=Time(250), min_count=6,
              max_count=9, name="borehole")
]
# /<--borehole-->


# <--pipe_and_power_network-->
# scaled by: scale_reqs
# measurement base: 1 km
PIPE_NETWORK = [
    WorkerReq(kind="driver", volume=Time(60), min_count=3,
              max_count=6, name="pipe_network"),
    WorkerReq(kind="fitter", volume=Time(60), min_count=3,
              max_count=6, name="pipe_network"),
    WorkerReq(kind="manager", volume=Time(20), min_count=2,
              max_count=4, name="pipe_network"),
    WorkerReq(kind="handyman", volume=Time(50), min_count=3,
              max_count=6, name="pipe_network"),
    WorkerReq(kind="electrician", volume=Time(20), min_count=2,
              max_count=4, name="pipe_network"),
]
POWER_NETWORK = scale_reqs(PIPE_NETWORK, scalar=1, new_name="power_network")
# /<--pipe_and_power_network-->
# <--flooding_light_mast-->
# scaled by: -
# measurement base: -
LIGHT_MAST = [
    WorkerReq(kind="driver", volume=Time(20), min_count=2,
              max_count=4, name="flooding_light_mast"),
    WorkerReq(kind="fitter", volume=Time(40), min_count=4,
              max_count=8, name="flooding_light_mast"),
    WorkerReq(kind="manager", volume=Time(10), min_count=2,
              max_count=4, name="flooding_light_mast"),
    WorkerReq(kind="handyman", volume=Time(60), min_count=3,
              max_count=12, name="flooding_light_mast"),
    WorkerReq(kind="electrician", volume=Time(5), min_count=1,
              max_count=1, name="flooding_light_mast"),
]
# /<--flooding_light_mast-->

# scaled by: mul_borehole_volume
# measurement base: tuple(volume_without_boreholes, volume_for_one_borehole)
HANDING_STAGE_BASE = (0.66, 0.02)
HANDING_STAGE = [
    WorkerReq(kind="driver", volume=Time(5), min_count=1,
              max_count=3, name="handing_stage"),
    WorkerReq(kind="manager", volume=Time(5), min_count=1,
              max_count=3, name="handing_stage"),
    WorkerReq(kind="engineer", volume=Time(10), min_count=2,
              max_count=8, name="handing_stage"),
]
