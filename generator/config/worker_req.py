from typing import List, Optional

from schemas.requirements import WorkerReq


def scale_reqs(req_list: List[WorkerReq], scalar: float, new_name: Optional[str] = None) -> List[WorkerReq]:
    return [work_req.scale(scalar, new_name) for work_req in req_list]


def mul_volume_reqs(req_list: List[WorkerReq], scalar: float, new_name: Optional[str] = None) -> List[WorkerReq]:
    return [work_req.mul_volume(scalar, new_name) for work_req in req_list]


def get_borehole_volume(borehole_count: int, base: (float, float)) -> float:
    return base[0] + base[1] * borehole_count


def mul_borehole_volume(req_list: List[WorkerReq], borehole_count: int, base: (float, float)) -> List[WorkerReq]:
    return mul_volume_reqs(req_list, get_borehole_volume(borehole_count, base))


START_PROJECT = []
END_PROJECT = []


# <--road_block-->
# scaled by: scale_reqs
# measurement base: 1 km
MIN_ROAD = [
    WorkerReq(type="driver", volume=15, min_count=3,
              max_count=9, name="min_road"),
    WorkerReq(type="manager", volume=5, min_count=1,
              max_count=3, name="min_road"),
    WorkerReq(type="handyman", volume=30, min_count=6,
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
    WorkerReq(type="driver", volume=900, min_count=6,
              max_count=30, name="engineering_preparation"),
    WorkerReq(type="manager", volume=150, min_count=2,
              max_count=6, name="engineering_preparation"),
    WorkerReq(type="handyman", volume=900, min_count=6,
              max_count=30, name="engineering_preparation"),
    WorkerReq(type="engineer", volume=150, min_count=2,
              max_count=6, name="engineering_preparation"),
]
# /<--engineering_preparation-->


# <--power_line-->
# scaled by: scale_reqs
# measurement base: 1 km
POWER_LINE = [
    WorkerReq(type="driver", volume=120, min_count=6,
              max_count=9, name="power_line"),
    WorkerReq(type="fitter", volume=120, min_count=6,
              max_count=9, name="power_line"),
    WorkerReq(type="manager", volume=40, min_count=2,
              max_count=6, name="power_line"),
    WorkerReq(type="handyman", volume=120, min_count=6,
              max_count=9, name="power_line"),
    WorkerReq(type="electrician", volume=40, min_count=2,
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
    WorkerReq(type="driver", volume=400, min_count=2,
              max_count=6, name="pipe_node"),
    WorkerReq(type="fitter", volume=800, min_count=4,
              max_count=8, name="pipe_node"),
    WorkerReq(type="manager", volume=400, min_count=2,
              max_count=8, name="pipe_node"),
    WorkerReq(type="handyman", volume=1200, min_count=6,
              max_count=12, name="pipe_node"),
    WorkerReq(type="electrician", volume=300, min_count=2,
              max_count=6, name="pipe_node"),
]
# /<--pipe_node-->


# <--metering_install-->
# scaled by: -
# measurement base: -
METERING_INSTALL = [
    WorkerReq(type="driver", volume=200, min_count=2,
              max_count=6, name="metering_install"),
    WorkerReq(type="fitter", volume=300, min_count=3,
              max_count=9, name="metering_install"),
    WorkerReq(type="manager", volume=50, min_count=1,
              max_count=2, name="metering_install"),
    WorkerReq(type="handyman", volume=300, min_count=4,
              max_count=12, name="metering_install"),
    WorkerReq(type="engineer", volume=100, min_count=2,
              max_count=4, name="metering_install"),
]
# /<--metering_install-->
# <--ktp_nep-->
# scaled by: mul_borehole_volume
# measurement base: tuple(volume_without_boreholes, volume_for_one_borehole)
KTP_NEP_BASE = (0.58, 0.03)
KTP_NEP = [
    WorkerReq(type="driver", volume=200, min_count=2,
              max_count=6, name="ktp_nep"),
    WorkerReq(type="fitter", volume=300, min_count=3,
              max_count=9, name="ktp_nep"),
    WorkerReq(type="manager", volume=50, min_count=1,
              max_count=2, name="ktp_nep"),
    WorkerReq(type="handyman", volume=300, min_count=4,
              max_count=12, name="ktp_nep"),
    WorkerReq(type="electrician", volume=100, min_count=2,
              max_count=4, name="ktp_nep"),
]
# <--ktp_nep-->
# scaled by: mul_volume_reqs
# measurement base: DRAINAGE_TANK_BASE_VOLUME + DRAINAGE_TANK_INPUT_VOLUME * borehole_count
# scaled by: mul_borehole_volume
# measurement base: tuple(volume_without_boreholes, volume_for_one_borehole)
DRAINAGE_TANK_BASE = (0.3, 0.05)
DRAINAGE_TANK = [
    WorkerReq(type="driver", volume=400, min_count=4,
              max_count=8, name="drainage_tank"),
    WorkerReq(type="manager", volume=50, min_count=1,
              max_count=2, name="drainage_tank"),
    WorkerReq(type="handyman", volume=300, min_count=3,
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
    WorkerReq(type="driver", volume=200, min_count=4,
              max_count=12, name="borehole"),
    WorkerReq(type="fitter", volume=150, min_count=3,
              max_count=9, name="borehole"),
    WorkerReq(type="manager", volume=25, min_count=1,
              max_count=3, name="borehole"),
    WorkerReq(type="handyman", volume=250, min_count=6,
              max_count=9, name="borehole")
]
# /<--borehole-->


# <--pipe_and_power_network-->
# scaled by: scale_reqs
# measurement base: 1 km
PIPE_NETWORK = [
    WorkerReq(type="driver", volume=60, min_count=3,
              max_count=6, name="pipe_network"),
    WorkerReq(type="fitter", volume=60, min_count=3,
              max_count=6, name="pipe_network"),
    WorkerReq(type="manager", volume=20, min_count=2,
              max_count=4, name="pipe_network"),
    WorkerReq(type="handyman", volume=50, min_count=3,
              max_count=6, name="pipe_network"),
    WorkerReq(type="electrician", volume=20, min_count=2,
              max_count=4, name="pipe_network"),
]
POWER_NETWORK = scale_reqs(PIPE_NETWORK, scalar=1, new_name="power_network")
# /<--pipe_and_power_network-->
# <--flooding_light_mast-->
# scaled by: -
# measurement base: -
LIGHT_MAST = [
    WorkerReq(type="driver", volume=20, min_count=2,
              max_count=4, name="flooding_light_mast"),
    WorkerReq(type="fitter", volume=40, min_count=4,
              max_count=8, name="flooding_light_mast"),
    WorkerReq(type="manager", volume=10, min_count=2,
              max_count=4, name="flooding_light_mast"),
    WorkerReq(type="handyman", volume=60, min_count=3,
              max_count=12, name="flooding_light_mast"),
    WorkerReq(type="electrician", volume=5, min_count=1,
              max_count=1, name="flooding_light_mast"),
]
# /<--flooding_light_mast-->

# scaled by: mul_borehole_volume
# measurement base: tuple(volume_without_boreholes, volume_for_one_borehole)
HANDING_STAGE_BASE = (0.66, 0.02)
HANDING_STAGE = [
    WorkerReq(type="driver", volume=5, min_count=1,
              max_count=3, name="handing_stage"),
    WorkerReq(type="manager", volume=5, min_count=1,
              max_count=3, name="handing_stage"),
    WorkerReq(type="engineer", volume=10, min_count=2,
              max_count=8, name="handing_stage"),
]
