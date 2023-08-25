from typing import List, Tuple, Union

import numpy as np

from optimizer import decode_locs, map_backward, split_solution


def logical_constrains(
    solution: List[Union[float, int]],
    n_opt_well: int,
    target: str
) -> Union[int, None]:
    """
    - Check if the two wells are located at the same location (if target is:loc) .
    - Check if the perforation interval is valid, start <= end (if target is: perf).

    > If a constraint is violated, return 1, else return None.

    Args:
        solution (list): The optimizer's decoded solution.
        n_opt_well (int): Number of wells being optimized.
        target (str): The target for applying constraints ('loc' or 'perf').

    Returns:
        int or None: Return 1 if a constraint is violated, else None.
    """
    j = 0
    well_item = []

    # Create a list of well locations/perforations
    for i in range(n_opt_well):
        well_item.append(list(solution[j:j + 2]))
        j += 2

    # Check if the two wells are located at the same location
    if target == 'loc':
        for n in range(len(well_item)):
            same_loc = well_item.count(well_item[n])
            if same_loc > 1:
                return 1

    # Check if the perforation interval is valid, start <= end
    if target == 'perf':
        for well_perf in well_item:
            if well_perf[0] > well_perf[1]:
                return 1

    return None


def loc_penalty(
    solution: List[Union[float, int]],
    target: str,
    n_optwell: int = 5,
    min_well_space: int = 2,
    obj_func: bool = False,
    grid_size: Tuple[int, int, int] = (20, 20, 5),
    best_rqi_locs=None
) -> Tuple[List[str], int]:
    """
    Penalize solutions violating well location constraints.

    Args:
        solution (list): The optimizer's decoded solution (in the obj_func) or GBS get from optimizer.
        target (str): The target for applying penalties ('min_space' or 'border').
        n_optwell (int, optional): Number of wells being optimized. Default is 5.
        min_well_space (int, optional): Minimum distance between wells. Default is 2.
        obj_func (bool, optional): Indicates if objective function is used. Default is False.
        grid_size (tuple, optional): Grid size for boundary checking. Default is (20, 20, 5).
        best_rqi_locs: This will be provided by another module.

    Returns:
        list: List of penalty messages.
        int: Total number of penalties.
    """
    result = []
    fault = 0

    # Calculate well locations from the solution
    if obj_func is False:
        # if not in obj_func, gbs must be decoded
        locs_id, prod_rate, inj_rate = split_solution(solution)
        well_locs = map_backward(locs_id, best_rqi_locs)
        locs, perfs = decode_locs(well_locs)
    else:
        locs = solution

    # Create a list of well locations
    well_item = [list(locs[j:j + 2]) for j in range(0, n_optwell * 2, 2)]

    # Check for minimum space between two wells
    if target == 'min_space':
        for well_number, well_i in enumerate(well_item):
            for next_index in range(well_number + 1, n_optwell):
                well_j = well_item[next_index]
                distance = np.sqrt(
                    np.square((well_i[0] - well_j[0])) + np.square((well_i[1] - well_j[1])))
                if distance < min_well_space:
                    result.append(
                        [f'> well {well_number + 1} & {next_index + 1}', f' distance = {round(distance, 3)}'])
                    fault += 1
        if not result:
            result.append(['Well spaces look good!'])

    # Check for boundary constraint (if well is located on boarders)
    if target == 'border':
        for well_num, well in enumerate(well_item, start=1):
            loc_i, loc_j = well
            if loc_i in (1, grid_size[0]) or loc_j in (1, grid_size[1]):
                result.append(
                    f'> Well {well_num}, is on the border --> i = {loc_i}, j = {loc_j}')
                fault += 1
        if not result:
            result.append(['No wells on the border!'])

    # Return penalty messages and total penalties
    return result, fault


def null_block_penalty(
    opt_solution: List[Union[float, int]],
    null_block_list: List[Tuple[int, int, int]],
    best_rqi_locs: Tuple[int, int, int],
    opt_well_number: int = 10,
    min_null_space: int = 2,
    obj_func: bool = False
) -> Tuple[List[str], int]:
    """
    Penalize solutions violating null block constraints.

    Args:
        null_block_list (list): List of null block coordinates.
        well_locs (list, optional): List of well locations. Default is None.
        well_perfs (list, optional): List of well perforations. Default is None.
        opt_well_number (int, optional): Number of wells being optimized. Default is 10.
        min_null_space (int, optional): Minimum distance to null blocks. Default is 2.
        opt_solution (list, optional): Optimizer's solution. Default is None.
        obj_func (bool, optional): Indicates if objective function is used. Default is False.

    Returns:
        list: List of penalty messages.
        int: Total number of penalties.
    """
    result = []
    total_fault = 0

    # Calculate well locations and perforations from the solution
    if obj_func is False:
        locs_id, _, _ = split_solution(opt_solution)
        locs = map_backward(locs_id, best_rqi_locs)
        well_locs, well_perfs = decode_locs(locs)

    locs_well = [list(well_locs[j:j + 2])
                 for j in range(0, opt_well_number * 2, 2)]
    perf_well = [list(well_perfs[j:j + 2])
                 for j in range(0, opt_well_number * 2, 2)]

    for well_number in range(opt_well_number):
        loc_i, loc_j = locs_well[well_number]
        perf_s, perf_e = perf_well[well_number]
        perf_c = perf_s + 1  # perf_center

        for null in null_block_list:
            null_i, null_j, null_z = null
            dis_sq_c = np.square(loc_i - null_i) + \
                np.square(loc_j - null_j) + np.square(perf_c)
            dis_c = np.sqrt(dis_sq_c)
            dis_sq_s = np.square(loc_i - null_i) + \
                np.square(loc_j - null_j) + np.square(perf_s)
            dis_s = np.sqrt(dis_sq_s)
            dis_sq_e = np.square(loc_i - null_i) + \
                np.square(loc_j - null_j) + np.square(perf_e)
            dis_e = np.sqrt(dis_sq_e)

            if null_i == loc_i and null_j == loc_j:
                if null_z in (perf_s, perf_c, perf_e):
                    result.append(
                        f'> Well {well_number + 1} is on forbidden location. Perforated on NULL BLOCK Grid of {null_z} --> Null Block: {null} ')
                    total_fault += 1
            else:
                if dis_s < min_null_space or dis_c < min_null_space or dis_e < min_null_space:
                    result.append(
                        f'> Well {well_number + 1} is on forbidden location. Well is near null block of {null} --> distance = {(dis_s, dis_c, dis_e)}')
                    total_fault += 1

    if not result:
        result.append(['No wells on Null block or near them!'])

    # Return penalty messages and total penalties
    return result, total_fault
