from pathlib import Path
from time import time
from typing import List, Tuple, Union

import numpy as np
from loguru import logger

from utils import path_check


def get_prop(
    file: str,
    keyword_list: list,
    size: tuple = (85, 185, 31),
    mode: str = 'matrix'
):
    """
    Read properties from a .GRID file and return them based on the specified mode.

    Args:
        file (str): Path to the .GRID file.
        keyword_list (list): List of property keywords to read from the .GRID file.

    Keyword Arguments:
        size (tuple): Reservoir grid size. Default is (85, 185, 31).
        mode (str): Output mode ('matrix', 'list', or 'both'). Default is 'matrix'.

    Returns:
        list or np.ndarray or tuple of both: Properties according to keyword_list and mode.
    """
    result_matrix = []
    result_list = []

    # Read file until it reachs to keyword and Do until / finds in line.
    for key in keyword_list:
        with open(file, 'r') as grid_file:

            # target is a list of lines after the keyword.
            target = []

            # i is a counter to skip the first line after the keyword.
            i = 0

            # Read file until it reachs to keyword.
            for line in grid_file:
                # If the line starts with the keyword, then read the next line.
                if line.startswith(key):
                    key_line = line
                    # Read file until it reachs to '/'.
                    while '/' not in key_line:
                        # Skip the first line after the keyword.
                        if i == 0:
                            next(grid_file)
                            key_line = next(grid_file)
                            target.append(key_line)
                            i += 1
                        else:
                            key_line = next(grid_file)
                            target.append(key_line)

        # that key should add to prop_list in type of list.
        prop_list = []
        for line in target:
            line_list = line.split()
            # Check if a property is same for several blocks and expressed as
            # 'number * value' (12 * 0.05)
            for num in line_list:
                if '/' not in num:
                    if '*' in num:
                        split_num = num.split("*")

                        if len(split_num) > 1:
                            times = split_num[0]
                            value = split_num[1]
                            # Add the value to the list for the number of times.
                            for i in range(int(times)):
                                # add the value to the list as float.
                                prop_list.append(float(value))
                        else:
                            # add the value to the list as float.
                            prop_list.append(float(split_num[0]))

                    # add the value to the list as float.
                    else:
                        prop_list.append(float(num))

        # that key should add to prop_list in type of matrix.
        prop_matrix = np.ones(size)
        index = 0
        for k in range(size[2]):
            for j in range(size[1]):
                for i in range(size[0]):
                    prop_matrix[i, j, k] = prop_list[index]
                    index += 1
        result_matrix.append(prop_matrix)
        result_list.append(prop_list)

    if mode == 'matrix':
        return result_matrix
    elif mode == 'list':
        return result_list
    else:
        return result_matrix, result_list


def possible_location(
    loc: tuple,
    null_space_lim: int = 2,
    grid_size: tuple = (85, 185, 31)
):
    """
    Determine possible locations based on given location and null space limit.

    Args:
        loc (tuple): Current location.

    Keyword Arguments:
        null_space_lim (int): Minimum distance to null blocks. Default is 2.
        grid_size (tuple): Reservoir grid size. Default is (85, 185, 31).

    Returns:
        tuple: Lower and upper boundaries of the search area.
    """
    limit = null_space_lim ** 2
    i, j, k = loc

    # Find the lower and upper boundaries of the search area.
    lower_x, upper_x = max(0, i - limit), min(grid_size[0] - 1, i + limit)
    lower_y, upper_y = max(0, j - limit), min(grid_size[1] - 1, j + limit)
    lower_z, upper_z = max(0, k - limit), min(grid_size[2] - 1, k + limit)

    return lower_x, upper_x, lower_y, upper_y, lower_z, upper_z


def find_null_blocks(
    prop_dic: dict,
    grid_size: tuple = (85, 185, 31)
):
    """
    Find null blocks in the property dictionary.

    Args:
        prop_dic (dict): Dictionary containing property values.

    Keyword Arguments:
        grid_size (tuple): Reservoir grid size. Default is (85, 185, 31).

    Returns:
        tuple: Lists of null block locations with index starting at 1 and 0.
    """
    null_loc = []
    null_loc_0 = []

    for k in range(grid_size[2]):
        for j in range(grid_size[1]):
            for i in range(grid_size[0]):

                # Check if all properties are 0. If so, add the location to the list (NULL BLUCK)
                if all(prop_dic[f'{key}'][i, j, k] == 0 for key in prop_dic.keys()):

                    # (index start with 1) this will use later to check optimized location whether they are near to null block.
                    null_loc.append((i + 1, j + 1, k + 1))

                    # this is null block which use for cleaning map. index will start by 0.
                    null_loc_0.append((i, j, k))

    return null_loc, null_loc_0


def rock_index(
    file: str,
    prop_key_list: list,
    grid_size=(85, 185, 31),
    threshold=1000
):
    """
    Calculate rock quality index and identify best RQI values and null blocks.

    Args:
        file (str): Path to the <.GRID> file.
        prop_key_list (list): List of property keywords for RQI calculation.

    Keyword Arguments:
        grid_size (tuple): Reservoir grid size. Default is (85, 185, 31).
        threshold (int): Minimum RQI threshold for filtering. Default is 1000.

    Returns:
        tuple: Best RQI values, best RQI locations, null block locations, null block locations with index starting at 0, RQI matrix.
    """
    # [1] Get properties
    result_list = get_prop(file, prop_key_list, grid_size, mode='matrix')
    prop_dic = {key: prop for key, prop in zip(prop_key_list, result_list)}

    # [2] Calculate RQI
    prop_values = [prop_dic[key] for key in prop_key_list]
    rock_quality_index = np.prod(prop_values, axis=0) * (1 - prop_dic['SWAT'])

    # [3] Slice up RQI matrix to Best RQI (list), Best RQI loc (list)
    indices = np.where((rock_quality_index >= threshold) & (
        rock_quality_index <= rock_quality_index.max()))
    best_rqi_ = rock_quality_index[indices]
    best_rqi_id_ = list(zip(*indices))

    # [4] Find null blocks based on prop_dictionary
    nulls, nulls_0 = find_null_blocks(prop_dic, grid_size)

    return best_rqi_, best_rqi_id_, nulls, nulls_0, rock_quality_index


def clean_borders(
    best_rqi: list, best_rqi_loc: list,
    grid_size: tuple = (85, 185, 31)
):
    """
    Clean best RQIs from border locations in x and y directions.

    Args:
        best_rqi (list): Best selected RQI values.
        best_rqi_loc (list): Locations of the best selected RQI values.

    Keyword Arguments:
        grid_size (tuple): Reservoir grid size. Default is (85, 185, 31).

    Returns:
        tuple: Cleaned no-border RQI values and their corresponding locations.
    """
    nb_rqi = []  # No border RQI
    nbrqi_index = []  # No border RQI locations

    for rqi, loc in zip(best_rqi, best_rqi_loc):
        x, y = loc[0], loc[1]
        if 0 < x < grid_size[0] - 1 and 0 < y < grid_size[1] - 1:
            nb_rqi.append(rqi)
            nbrqi_index.append(loc)

    return nb_rqi, nbrqi_index


def clean_nulls(
    null_locs: list,
    best_rqi: list, rqi_locs: list,
    min_space: int = 2,
    grid_dim: tuple = (85, 185, 31)
):
    """
    Clean RQI locations by ensuring a minimum distance from null blocks.

    Args:
        null_locs (list): List of null block locations (index starts at 0).
        best_rqi (list): Best RQI values.
        rqi_locs (list): Best RQI locations.

    Keyword Arguments:
        min_space (int): Minimum available distance from null blocks. Default is 2.
        grid_dim (tuple): Reservoir grid size. Default is (85, 185, 31).

    Returns:
        tuple: Cleaned RQI values and their corresponding locations.
    """
    rqi_clean_blocks = []
    clean_rqi = []

    # Create a zeros matrix and mark null block locations with 1.
    null_zero_one_matrix = np.zeros(grid_dim)
    for null in null_locs:
        i, j, k = null
        null_zero_one_matrix[i, j, k] = 1

    for idx, sel_loc in enumerate(rqi_locs):
        i, j, k = sel_loc
        distance_passed = True  # Flag to track null block distance condition
        for x in range(i - min_space, i + min_space + 1):
            if not (0 <= x < grid_dim[0]):
                continue
            for y in range(j - min_space, j + min_space + 1):
                if not (0 <= y < grid_dim[1]):
                    continue
                for z in range(k - min_space, k + min_space + 1):
                    if not (0 <= z < grid_dim[2]):
                        continue
                    if null_zero_one_matrix[x, y, z] == 1:
                        dist_sq = (i - x) ** 2 + (j - y) ** 2 + (k - z) ** 2
                        distance = dist_sq ** 0.5
                        if distance < min_space:
                            distance_passed = False
                            break
                if not distance_passed:
                    break
            if not distance_passed:
                break
        if distance_passed:
            rqi_clean_blocks.append(sel_loc)
            clean_rqi.append(best_rqi[idx])

    return clean_rqi, rqi_clean_blocks


def clean_interval(
    rqi_loc_list: List[Tuple[int, int, int]],
    rqi_list: List[float],
    rqi_matrix: np.ndarray,
    threshold: float = 1000
) -> Tuple[List[Tuple[int, int, int]],
           List[float],
           List[Tuple[int, int, int, int]],
           List[float]]:
    """
    Clean and filter RQI values based on intervals.

    Args:
        rqi_loc_list (List[Tuple[int, int, int]]): List of RQI locations.
        rqi_list (List[float]): List of RQI values.
        rqi_matrix (np.ndarray): 3D matrix containing RQI values.
        threshold (float, optional): Threshold for RQI filtering. Defaults to 1000.

    Returns:
        Tuple[List[Tuple[int, int, int]],
              List[float],
              List[Tuple[int, int, int, int]],
              List[float]]: Filtered RQI locations, RQI values, interval locations, and interval RQI values.
    """

    final_best_loc = []
    final_best_rqi = []
    ok_interval_loc = []
    ok_interval_rqi = []

    for idx, loc in enumerate(rqi_loc_list):
        x, y, z_c = loc
        z_s, z_e = z_interval(z_c, grid_size=rqi_matrix.shape)
        rqi_s, rqi_c, rqi_e = rqi_matrix[x, y,
                                         z_s], rqi_matrix[x, y, z_c], rqi_matrix[x, y, z_e]

        if all(rqi >= threshold for rqi in [rqi_s, rqi_c, rqi_e]):
            interval_loc = (x, y, z_s, z_e)
            interval_rqi = np.average([rqi_s, rqi_c, rqi_e])
            final_best_loc.append(loc)
            final_best_rqi.append(rqi_list[idx])
            ok_interval_loc.append(interval_loc)
            ok_interval_rqi.append(interval_rqi)

    return final_best_loc, final_best_rqi, ok_interval_loc, ok_interval_rqi


def z_interval(center_block: int,
               grid_size: Tuple[int, int, int],
               interval: int = 2) -> Tuple[int, int]:
    z_s = max(center_block - interval, 0)
    z_e = min(center_block + interval, grid_size[-1] - 1)
    return z_s, z_e


def filter_locs(
    file: str,
    prop_key_list: List[str],
    null_space: int = 2,
    threshold: int = 1000,
    grid_size: Tuple[int, int, int] = (85, 185, 31),
    check_interval: bool = True
) -> Union[
    Tuple[List[float], List[Tuple[int, int, int]],
          List[Tuple[int, int, int]], List[float], List[Tuple[int, int, int]]],
        Tuple[List[float], List[Tuple[int, int, int]], List[Tuple[int, int, int]]]]:
    """
    This function performs various filtering and cleaning operations on RQI (Rock Quality Index) data.
    It reads properties from a <.GRID> file, calculates RQI, filters null blocks and border locations,
    and checks distances to null blocks. Finally, it returns the filtered RQI values, their locations,
    null block locations, and final RQI values/locations.

    Args:
        file (str): Path to the <.GRID> file.
        prop_key_list (List[str]): A list of properties keywords (according to the .GRID file).

    Keyword Args:
        null_space (int): The minimum distance to null blocks. (default: 2)
        threshold (int): Criteria for filtering. (default: 1000)
        grid_size (Tuple[int, int, int]): Dimensions of the grid. (default: (85, 185, 31))
        check_interval (bool): Whether to perform interval checking. (default: True)

    Returns:
        Tuple: Depending on check_interval, returns a tuple of four or three lists:
        - filtered RQI values
        - filtered RQI locations
        - null block locations
        - final RQI values/locations
    """
    # Calculate initial RQI values and locations
    best_rqi, best_rqi_locs, null_locs, null_0_locs, rqi_matrix = rock_index(
        file, prop_key_list, grid_size, threshold)

    # Clean borders and get non-border RQI values/locations
    nb_rqi, nb_rqi_locs = clean_borders(best_rqi, best_rqi_locs, grid_size)

    # Clean nulls and get final RQI values/locations
    final_rqi, final_rqi_locs = clean_nulls(null_0_locs, nb_rqi, nb_rqi_locs, min_space=null_space,
                                            grid_dim=grid_size)

    if check_interval:
        # Clean intervals and get final RQI values/locations after interval checking
        final_rqi_locs_, final_rqi_, _, _ = clean_interval(
            final_rqi_locs, final_rqi, rqi_matrix, TH=threshold)
        return final_rqi, final_rqi_locs, null_locs, final_rqi_, final_rqi_locs_
    else:
        return final_rqi, final_rqi_locs, null_locs


def main(
    rqi_limit=0,
    grid_path='INCLUDE/SSA_RQI_0.GRDECL',
    null_space=2,
    prop_key=['PERMX', 'PORO', 'NTG', 'TRANY', 'TRANZ', 'SWAT'],
    dim=(85, 185, 31)
):
    """
    Main function to process RQI data.

    Args:
        rqi_limit (int, optional): RQI limit. Defaults to 0.
        grid_path (str, optional): Path to the <.GRID> file. Defaults to 'INCLUDE/SSA_RQI_0.GRDECL'.
        null_space (int, optional): Minimum distance to null blocks. Defaults to 2.
        prop_key (List[str], optional): List of properties keywords. Defaults to ['PERMX', 'PORO', 'NTG', 'TRANY', 'TRANZ', 'SWAT'].
        dim (Tuple[int, int, int], optional): Dimensions of the grid. Defaults to (85, 185, 31).
    """

    # Record the start time of the program.
    start_time = time()

    # Inputs
    rqi_limit = 0

    abs_path = Path(__file__).resolve().parent.parent
    grid_path = f'{abs_path}/{grid_path}'

    # RQI files will be saved here.
    rqi_path = f'{abs_path}/RQI/TH={rqi_limit}'

    # RQI map considering interval check will save here
    interval_path = f'{rqi_path}/interval_checked'
    path_check(interval_path)  # if path doesn't exist, create it.

    # Filter locations
    best_rqi, best_rqi_locs, null_blocks, rqi_interval, locs_interval = filter_locs(
        grid_path, prop_key, null_space, rqi_limit, dim, check_interval=True)

    # Write answers using loguru logger
    with open(f'{rqi_path}/best_RQI.text', 'a+') as rqi_file:
        rqi_file.write('\n'.join(str(rqi) for rqi in best_rqi))

    with open(f'{rqi_path}/best_RQI_locs.text', 'a+') as rqi_locs_file:
        for item in best_rqi_locs:
            x, y, z = item[0], item[1], item[2]
            rqi_locs_file.write(f'{x}, {y}, {z}\n')

    with open(f'{rqi_path}/Null_blocks.text', 'a+') as null_file:
        for item in null_blocks:
            x, y, z = item[0], item[1], item[2]
            null_file.write(f'{x}, {y}, {z}\n')

    with open(f'{interval_path}/best_RQI.text', 'a+') as rqi_interval_file:
        rqi_interval_file.write('\n'.join(str(rqi) for rqi in rqi_interval))

    with open(f'{interval_path}/best_RQI_locs.text', 'a+') as locs_interval_file:
        for item in locs_interval:
            x, y, z = item[0], item[1], item[2]
            locs_interval_file.write(f'{x}, {y}, {z}\n')

    # Turn off timer
    end_time = time()
    run_time = round(abs(end_time - start_time) / 60, 3)

    # Log completion
    logger.info(f'Process is finished!\n Runtime: {run_time} min.')


if __name__ == "__main__":

    rqi_limit = 0
    grid_path = 'INCLUDE/SSA_RQI_0.GRDECL'
    null_space = 2
    prop_key = ['PERMX', 'PORO', 'NTG', 'TRANY', 'TRANZ', 'SWAT']
    dim = (85, 185, 31)

    main(
        rqi_limit=rqi_limit,
        grid_path=grid_path,
        null_space=null_space,
        prop_key=prop_key,
        dim=dim
    )
