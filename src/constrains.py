from typing import List, Union, Tuple

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
    solution: Union[List[Union[float, int]], Tuple[float, ...]],
    target: str,
    n_optwell: int = 5,
    min_well_space: int = 2,
    obj_func: bool = False,
    grid_size: Tuple[int, int, int] = (20, 20, 5)
) -> Tuple[List[Union[str, List[str]]], int]:
    """
    Calculate penalties based on the well locations.

    Args:
        solution (list or tuple): The optimizer's decoded solution.
        target (str): The target for applying penalties ('min_space' or 'border').
        n_optwell (int, optional): Number of wells being optimized. Default is 5.
        min_well_space (int, optional): Minimum allowed well spacing. Default is 2.
        obj_func (bool, optional): Whether the solution is for the objective function. Default is False.
        grid_size (tuple, optional): Size of the grid. Default is (20, 20, 5).

    Returns:
        tuple: A tuple containing a list of penalty information and the total number of penalties.
    """
    # Check if the solution is for the objective function or not
    if obj_func is False:
        locs_id, prod_rate, inj_rate = split_solution(solution)
        well_locs = map_backward(locs_id, best_rqi_locs)
        locs, perfs = decode_locs(well_locs)
    else:
        locs = solution
    
    j = 0
    well_item = []
    result = []
    fault = 0
    
    # Create a list of well locations
    for _ in range(n_optwell):
        well_item.append(list(locs[j:j + 2]))
        j += 2
    
    if target == 'min_space':
        for well_number in range(n_optwell):
            well_i = well_item[well_number]
            next_index = well_number + 1
            while next_index < n_optwell:
                well_j = well_item[next_index]
                x_distance = np.square((well_i[0] - well_j[0]))
                y_distance = np.square((well_i[1] - well_j[1]))
                dist = np.sqrt(x_distance + y_distance)
                if dist < min_well_space:
                    result.append([f'> well {well_number + 1} & {next_index + 1}', f' distance = {round(dist, 3)}'])
                    fault += 1
                    next_index += 1
                else:
                    next_index += 1
        if len(result) == 0:
            result.append(['Well spacing looks good!'])
    
    if target == 'border':
        well_num = 0
        for well in well_item:
            well_num += 1
            loc_i, loc_j = well
            if loc_i in (1, grid_size[0]) or loc_j in (1, grid_size[1]):
                result.append([f'> Well {well_num}, is on border --> i = {loc_i}, j = {loc_j}'])
                fault += 1
        if len(result) == 0:
            result.append(['No well on the border!'])
    
    return result, fault