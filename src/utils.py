import subprocess
from pathlib import Path
from typing import List, Union

import numpy as np
from loguru import logger

from optimizer import SSA_WP


def path_check(file_path: Union[str, Path]) -> None:
    """
    Check if the given path exists and create it if it doesn't.

    Args:
        file_path (str or Path): The path to be checked and created if not exist.
    """
    # Create a Path object from the given file_path
    path = Path(file_path)

    # Check if the path exists
    if not path.exists():
        # Create the directory, including parent directories if necessary
        path.mkdir(parents=True, exist_ok=True)
        
        # Log the creation of the directory
        logger.info(f"Created directory: {path}")


def solution_track(
    content: List[str],
    filename: str = 'track_locs.txt'
) -> None:
    """
    Append the provided solution content to a specified file, creating the file and directory if they don't exist.

    This function is designed to track various solution packs during optimization, including:
    1. Production and injection locations
    2. Perforation intervals
    3. Injection rates
    4. Production rates

    Args:
        content (list or array): The solution content to be tracked.

    Keyword Args:
        filename (str): The name of the file to store the tracking data. Default is 'track_locs.txt'.
    """
    # Define the directory path to store tracking files
    path = 'Log_Files/Track_solutions'
    
    # Ensure the directory exists or create it
    path_check(path)
    
    # Create the full file path
    file_path = f'{path}/{filename}'
    
    # Append the solution content to the tracking file
    with open(file_path, 'a') as track_log:
        track_log.write('Obj function Called \n')
        for line in content:
            track_log.write(f'{line}\n')
        track_log.write('------------------------------------------------------\n')
    
    logger.info(f"Track solutions: Appended solution to {file_path}")


def prepare_welspecs_content(
    solution: Union[List[Union[float, int]], np.ndarray], 
    n_prod_well: int, n_inj_well: int
)-> List[str]:
    """
    Prepare content for the WELSPECS section of `.DATA` file based on the given solution.  
    
    Args:
        solution (list or array): The optimizer's decoded solution.
        n_prod_well (int): Number of production wells being optimized.
        n_inj_well (int): Number of injection wells under optimization.

    Returns:
        list: Content lines for the WELSPECS section.
        
    The template considered for well location is as follows:

    Production wells: [' ', well_name, 'PRD', loc_i, loc_j, '1*', 'OIL/']
    
    Injection wells: [' ', well_name, 'INJ', loc_inj_i, loc_inj_j, '1*', 'WATER/']
    
    ~ Example of solution:
    
    solution: [1, 10, ..., 2, 8, ...] which is interpreted as: i_p=1, j_p=10, i_i=2, j_i=8.
    """
    content = []
    j = 0
    
    for w_p in range(1, n_prod_well + 1):
        # well name
        pw_name = f'P{w_p}'
        # well locations (i, j)
        loc_i = str(int(solution[j]))
        loc_j = str(int(solution[j + 1]))

        # Create production well content
        pw_content = ['', pw_name, 'PRD', loc_i, loc_j, '1*', 'OIL/']
        temp_list = '   '.join(pw_content)
        content.append(temp_list)
        j += 2
        
    for w_i in range(1, n_inj_well + 1):
        # well name
        iw_name = f'I{w_i}'
        # well locations (i, j)
        loc_inj_i = str(int(solution[j]))
        loc_inj_j = str(int(solution[j + 1]))

        # Create injection well content
        iw_content = ['', iw_name, 'INJ', loc_inj_i, loc_inj_j, '1*', 'WATER/']
        temp_list = '   '.join(iw_content)
        content.append(temp_list)
        j += 2
    
    return content


def prepare_compdat_content(
    solution: Union[List[Union[float, int]], np.ndarray],
    n_prod_well: int, n_inj_well: int
) -> List[str]:
    """
    Prepare content for the COMPDAT section of `.DATA` file based on the given solution.

    Args:
        solution (list or array): The optimizer's decoded solution.
        n_prod_well (int): Number of production wells being optimized.
        n_inj_well (int): Number of injection wells under optimization.

    Returns:
        list: Content lines for the COMPDAT section.

    The template considered for well perforation is as follows:

    Production wells: ['', pro_name, '2*', pw_perf_s, pw_perf_e, 'OPEN', '2*', '0.5/']
    
    Injection wells: ['', inj_name, '2*', wi_perf_s, wi_perf_e, 'OPEN', '2*', '0.5/']
    
    ~ Example of solution:
    
    solution: [1, 10, ..., 2, 8, ...] which is interpreted as: pw_perf_s=1, pw_perf_e=10, wi_perf_s=2, wi_perf_e=8.
    """
    content = []
    p = 0
    
    for w_p in range(1, n_prod_well + 1):
        # well name
        pw_name = f'P{w_p}'
        # well perforation start and end
        pw_perf_s = str(int(solution[p]))
        pw_perf_e = str(int(solution[p + 1]))

        # Create production well perforation content
        pw_content = ['', pw_name, '2*', pw_perf_s, pw_perf_e, 'OPEN', '2*', '0.5/']
        temp_list = '   '.join(pw_content)
        content.append(temp_list)
        p += 2

    for w_i in range(1, n_inj_well + 1):
        # well name
        iw_name = f'I{w_i}'
        # well perforation start and end
        wi_perf_s = str(int(solution[p]))
        wi_perf_e = str(int(solution[p + 1]))

        # Create injection well perforation content
        iw_content = ['', iw_name, '2*', wi_perf_s, wi_perf_e, 'OPEN', '2*', '0.5/']
        temp_list = '   '.join(iw_content)
        content.append(temp_list)
        p += 2

    return content


def prepare_wconprod_content(
    solution: Union[List[Union[float, int]], np.ndarray],
    n_prod_well: int
) -> List[str]:
    """
    Prepare content for the WCONPROD section of `.DATA` file based on the given solution.

    Args:
        solution (list or array): The optimizer's decoded solution.
        n_prod_well (int): Number of production wells being optimized.

    Returns:
        list: Content lines for the WCONPROD section.

    The template considered for production well rates is as follows:

    Production wells: ['', well_name, 'OPEN', 'ORAT', prod_rate, '4*', '2000/']
    
    ~ Example of solution:
    
    solution: [100, 200] which is interpreted as: prod_rate_w1=100, prod_rate_w2=200.
    """
    content = []
    z = 0
    for i in range(1, n_prod_well + 1):
        # well name
        well_name = f'P{i}'
        # production rate
        prod_rate = str(solution[z])

        # Create production well rate content
        pr_content = ['', well_name, 'OPEN', 'ORAT', prod_rate, '4*', '2000/']
        temp_list = '   '.join(pr_content)
        content.append(temp_list)
        z += 1

    return content

def prepare_wconinje_content(
    solution: Union[List[Union[float, int]], np.ndarray],
    n_inj_well: int
) -> List[str]:
    """
    Prepare content for the WCONINJE section of `.DATA` file based on the given solution.

    Args:
        solution (list or array): The optimizer's decoded solution.
        n_inj_well (int): Number of injection wells under optimization.

    Returns:
        list: Content lines for the WCONINJE section.

    The template considered for injection well rates is as follows:

    Injection wells: ['', well_name, 'WAT', 'OPEN', 'RATE', inj_rate, '1*', '15000/']
    
    ~ Example of solution:
    
    solution: [500, 800] which is interpreted as: inj_rate_w1=500, inj_rate_w2=800.
    """
    content = []
    inj_idx = 0
    for i in range(1, n_inj_well + 1):
        # well name
        well_name = f'I{i}'
        # injection rate
        inj_rate = str(solution[inj_idx])

        # Create injection well rate content
        ir_content = ['', well_name, 'WAT', 'OPEN', 'RATE', inj_rate, '1*', '15000/']
        temp_list = '   '.join(ir_content)
        content.append(temp_list)
        inj_idx += 1

    return content


def write_to_file(
    file_path: str, 
    content: List[str]
) -> None:
    """
    Write the given content to a file.

    Args:
        file_path (str): The path to the file where the content will be written.
        content (list): The list of lines to be written to the file.

    Returns:
        None
    """
    try:
        with open(file_path, 'w+') as file:
            file.write('\n'.join(content) + '\n/')
    except IOError as e:
        logger.error(f"An error occurred while writing to the file: {e}")


def write_solution(
    solution: Union[List[Union[float, int]], np.ndarray],
    keyword: str,
    n_prod_well: int,
    n_inj_well: int,
    copy: bool = False,
    track: bool = False
) -> None:
    """
    Prepare a text file which includes optimizer solution called (INCLUDE) in .DATA file.

    There are four structures (templates):
    [1] Production and Injection locations (WELSPECS);
    [2] Production and Injection perforation (COMPDAT);
    [3] Production well rates (WCONPROD);
    [4] Injection well rates (WCONINJE).

    Args:
        solution (list or array): The solution proposed by the optimizer.
        keyword (str): Relevant keyword.
        n_prod_well (int): Number of production wells being optimized.
        n_inj_well (int): Number of injection wells under optimization.

    Keyword Args:
        copy (bool): This keyword is just for the best solution (default: False).
        track (bool): Whether to track the solution (default: False).
    """
    logger.info(f"Writing solution for '{keyword}' keyword.")
    
    FILENAME_EXTENSION = '.inc'
    INCLUDE_PATH = 'INCLUDE'
    BEST_SOLUTION_PATH = 'Log_Files/best_solution_files'

    if copy:
        # At the end of optimization, copy best solution files to a specific directory.
        output_path = BEST_SOLUTION_PATH
    else:
        # All the time, write the content in `INCLUDE` directory to read by `.DATA` file.
        output_path = INCLUDE_PATH

    file_name = keyword.lower()
    file_path = f'{output_path}/{file_name}{FILENAME_EXTENSION}'

    if keyword == 'WELSPECS':
        content = prepare_welspecs_content(solution, n_prod_well, n_inj_well)

    elif keyword == 'COMPDAT':
        content = prepare_compdat_content(solution, n_prod_well, n_inj_well)

    elif keyword == 'WCONPROD':
        content = prepare_wconprod_content(solution, n_prod_well)

    elif keyword == 'WCONINJE':
        content = prepare_wconinje_content(solution, n_inj_well, track=track)

    logger.debug(f"Writing content to file: {file_path}")
    write_to_file(file_path, content)
    
    if track:
        track_file_name = f'track_{file_name}.txt'
        solution_track(content, filename=track_file_name)


def batch_summary() -> None:
    """
    Summarize the occurrences of specific keywords in a batch log file in whole optimization process.
    keywords: Errors, Problems, Warnings.

    Returns:
        None
    """
    # Create a directory to store the summary file
    log_dir = 'Log_Files'
    path_check(log_dir)
    
    # Keywords to loop over
    keywords = ['Errors', 'Problems', 'Warnings']
    
    # Loop over the keywords and modes
    for keyword in keywords:
        mode = 'w+' if keyword == 'Errors' else 'a'
        
        call_counter = 0
        total_keyword = 0
        
        # Open the batch log file and count the occurrences of the keyword
        with open(f'{log_dir}/bat_results.txt', 'r') as bf_log, open(f'{log_dir}/bat_summary.txt', mode) as summary:
            for line in bf_log:
                if keyword in line:
                    call_counter += 1
                    
                    # Split the line into a list of strings
                    error_list = line.split()
                    
                    # How many times the keyword was found in the current simulation
                    error_count = int(error_list[1])
                    
                    if error_count != 0:
                        total_keyword += error_count
                        summary.write(
                            f'>> During "{call_counter}" Call ---> "{error_count}" {keyword} found.\n')
            
            summary.write(f'>>> Total {keyword}: {total_keyword}\n')
            if mode != 'a':
                summary.write(f'>>> >>> Objective function (or batch file) was called {call_counter} times in total.\n')
            summary.write('---------------------------------------------------------------------------\n')


def info_classify(key: str) -> None:
    """
    Classify and summarize simulation info lines based on a keyword.

    Args:
        key (str): The keyword used to identify the info type.
    """
    # Convert the keyword to uppercase
    key = key.upper()
    key_lines = []
    
    # Open the info file and extract the lines containing the keyword
    with open(f'Log_Files/Simulation_info/{key}_info.txt', 'r') as sim_info_file:
        for line in sim_info_file:
            if ' @--' in line:
                line = sim_info_file.readline()
                key_lines.append(line)
    
    key_line_set = list(set(key_lines))
    items = []
    occ_list = []
    
    for key_line in key_line_set:
        items.append(key_line)
        occ_list.append(key_lines.count(key_line))
    
    with open('Log_Files/Simulation_info/info_type_class.txt', 'a') as info_type_class:
        for i, item in enumerate(items):
            info_type_class.write(f'> {key}: {item}')
            info_type_class.write(f'>> occurrence: {occ_list[i]}\n')
            info_type_class.write('-------------------------------------\n')


def get_simulation_info(key_list: List[str] = None) -> None:
    """
    Extract and categorize simulation info based on a list of keywords.

    Args:
        key_list (List[str], optional): A list of keywords to extract info for. Default is ['PROBLEM', 'WARNING', 'ERROR'].
    """
    if key_list is None:
        key_list = ['PROBLEM', 'WARNING', 'ERROR']
    
    info_dir = 'Log_Files/Simulation_info'
    path_check(info_dir)
    
    for key in key_list:
        counter = 0
        key_problem = 0
        
        with open('Log_Files/bat_results.txt', 'r') as bat_file:
            for line in bat_file:
                if '(base)' in line:
                    counter += 1
                if f'@--{key}' in line:
                    key_problem += 1
                    with open(f'{info_dir}/{key}_info.txt', 'a') as k_info_file:
                        k_info_file.write(f'---------------------------------------------------------------\n')
                        k_info_file.write(f'Call {counter}\n')
                        k_info_file.write(f'---------------------------------------------------------------\n')
                    state = True
                    while state:
                        with open(f'{info_dir}/{key}_info.txt', 'a') as k_info_file:
                            k_info_file.write(f'{line}')
                        line = bat_file.readline()
                        if '@' not in line:
                            state = False
            else:
                if key_problem == 0:
                    with open(f'{info_dir}/{key}_info.txt', 'a') as k_info_file:
                        k_info_file.write(f"There's no {key} in the batch summary file.\n")
        info_classify(key)


def final_idx_check(
    idx: int, 
    len_obj_map: int
) -> int:
    """
    Check if the index chosen by the optimizer is within the bounds of the RQI map list.
    If not, adjust it to be within the valid range. This is necessary due to potential
    out-of-range issues caused by discrete adjustments in the solution.

    Args:
        idx (int): The index proposed by the optimizer.
        len_obj_map (int): The length of the RQI map list.

    Returns:
        int: An acceptable index within the range of the RQI map list length.
    """
    lb, ub = 0, len_obj_map - 1

    # Check if the index is outside the valid range and adjust it
    if idx not in range(lb, ub + 1):
        if idx >= ub:
            return ub
        if idx <= lb:
            return lb
    else:
        return idx


def ssa_log(
    filename: str, 
    solution: List, 
    npv_unit: str = 'M'
):
    """
    Log the best solution of optimizer with its hyper-parameters to a file.

    Args:
        filename (str): Name of the log file which is opened by mealpy.
        solution (list): Solution containing optimization details.
        npv_unit (str, optional): NPV unit. Defaults to 'M'.
    """
    # Save the results in Log_Files directory
    path = f'Log_Files'
    # Check if the path exists
    path_check(path)
    
    file_path = f'{path}/{filename}'

    # Extract the solution details 
    time = solution[0]              # Total Runtime
    best_solution = solution[1]     # Best Global Solution
    best_npv = solution[2]          # NPV of the best solution
    # Extract the SSA hyper parameters
    PD, SD, ST, epoch, pop_size = solution[3], solution[4], solution[5], solution[6], solution[7]

    # Append the information to the end of log file
    with open(file_path, 'a') as log_file:
            log_file.write(f'> Best solution: {best_solution}\n> Best NPV: ${best_npv} {npv_unit}\n')
            log_file.write(f'> PD: {PD};  SD: {SD};  ST:{ST};  Pop_size: {pop_size};  Epoch: {epoch}\n')
            log_file.write(
                f'> Runtime:{str(round(time, 2))} sec. ---> {str(round(time / 60, 2))} min ----> {str(round(time / 60 / 60, 2))} h\n')
            log_file.write(
                '-----------------------------------------------------------------------------------------------------------\n')

    logger.info(f'{filename} is ready!')


def final_result(
    solution: List, 
    capex_value: float, 
    npv_unit: str = 'M', 
    n_pro_well: int = 30, n_optimum_well: int = 36,
    well_space: int = 2,
    key_dic: dict = None, 
    best_rqi_locs = None,
    dims = (85, 185, 31),
    spaces: List = None, border: List = None, null: List = None
):
    """
    Log the final optimization result to a file.

    Args:
        solution (list): Solution containing optimization details.
        capex_value (float): Capex value.
        n_pro_well (int): Number of production wells.
        n_optimum_well (int, optional): Number of optimum wells. Defaults to 5.
        npv_unit (str, optional): NPV unit. Defaults to 'M'.
        spaces (list, optional): List of well spaces. Defaults to None.
        border (list, optional): List of wells on borders. Defaults to None.
        null (list, optional): List of wells near null blocks. Defaults to None.
    """
    path = f'Log_Files'
    path_check(path)  # You need to implement path_check function
    path_file = f'{path}/Best_Result.txt'
    time = solution[0]
    best_solution = solution[1]
    best_npv = solution[2]
    PD, SD, ST, epoch, pop_size = solution[3], solution[4], solution[5], solution[6], solution[7]

    wp_obj = SSA_WP(
        prod_well=n_pro_well, inj_well=n_optimum_well - n_pro_well, 
        well_space=well_space,
        dims=dims,
        best_rqi_locs=best_rqi_locs,
        keys=key_dic,
        capex=capex_value,
    )
    
    # Calculate well locations and perform other necessary tasks using imported functions
    locs_id, prod_rate, inj_rate = wp_obj.split_solution(best_solution)
    well_locs = wp_obj.map_backward(locs_id)
    locs, perfs = wp_obj.decode_locs(well_locs)
    
    # write the decoded solution (locations) of optimizer 
    write_solution(
        locs, 
        keyword=key_dic['loc_key'], 
        n_prod_well=n_pro_well, n_inj_well=n_optimum_well - n_pro_well
    )
    
    # write the decoded solution (perforations) of optimizer 
    write_solution(
        locs, 
        keyword=key_dic['perf_key'], 
        n_prod_well=n_pro_well, n_inj_well=n_optimum_well - n_pro_well
    )
    
    # write the decoded solution (production rates) of optimizer 
    write_solution(
        locs, 
        keyword=key_dic['pro_rate_key'], 
        n_prod_well=n_pro_well, n_inj_well=n_optimum_well - n_pro_well
    )

    # write the decoded solution (inj rates) of optimizer 
    write_solution(
        locs, 
        keyword=key_dic['inj_rate_key'], 
        n_prod_well=n_pro_well, n_inj_well=n_optimum_well - n_pro_well
    )

    # Write the final solution to a file
    j = 0
    rp = 0
    with open(path_file, 'w+') as final_text:
        final_text.write('>>> The final solution is according to below:\n')
        for i in range(0, n_optimum_well):
            # well locations (i, j)
            loc_i = locs[j]
            loc_j = locs[j + 1]
            
            # well perforation start and end
            perf_i = perfs[j]
            perf_j = perfs[j + 1]
            
            # separation between production and injection rates
            if rp in range(n_pro_well):
                q = str(prod_rate[rp])
                rp += 1
                ri = rp - n_pro_well
            else:
                q = str(inj_rate[ri])
                ri += 1
            
            # Write the well details to the file
            final_text.write(
                f'> well {i + 1} ---> i = {loc_i}, j = {loc_j}, perf_s = {perf_i}, perf_e = {perf_j}, Q = {q} STB\n')
            j += 2

        # Write the best solution and its NPV to the file (final details)
        final_text.write('----------------------------------------------------------------- \n')
        final_text.write('>>> The Best NPV obtained by this solution is:\n')
        final_text.write(f'$ {str(best_npv)} {npv_unit}  by capex value of {capex_value}\n')
        final_text.write('----------------------------------------------------------------- \n')
        final_text.write('>>> The SSA parameters:\n')
        final_text.write(f'> PD: {PD};  SD: {SD};  ST:{ST};  Pop_size: {pop_size};  Epoch: {epoch}\n')
        final_text.write(f'> Runtime: {str(round(time, 2))} sec. --> {str(round(time / 60, 2))} min --> {str(round(time / 60 / 60, 2))} h\n')
        final_text.write('\n ----------------------------------------------------------- \n')

        # Log spaces, borders, and null blocks if available
        if spaces is not None:
            for well in spaces:
                final_text.write(f'{well}\n')
        final_text.write('\n ----------------------------------------------------------- \n')
        
        if border is not None:
            for well in border:
                final_text.write(f'{well}\n')
        final_text.write('\n ----------------------------------------------------------- \n')
        
        if null is not None:
            for well in null:
                final_text.write(f'{well}\n')
        final_text.write('\n ----------------------------------------------------------- \n')

    # Run the simulation using the best solution
    with open(path_file, 'a') as bf_result:
        subprocess.call([r"$MatEcl.bat"], stdout=bf_result)
        bf_result.write('\n -----------------------------------------------------------\n')
        
    
    # Copy the best solution files to a specific directory 
    write_solution(
        locs, 
        keyword=key_dic['loc_key'], 
        n_prod_well=n_pro_well, n_inj_well=n_optimum_well - n_pro_well,
        copy=True
    )
    
    # Copy the best solution files to a specific directory 
    write_solution(
        locs, 
        keyword=key_dic['perf_key'], 
        n_prod_well=n_pro_well, n_inj_well=n_optimum_well - n_pro_well,
        copy=True
    )
    
    # Copy the best solution files to a specific directory 
    write_solution(
        locs, 
        keyword=key_dic['pro_rate_key'], 
        n_prod_well=n_pro_well, n_inj_well=n_optimum_well - n_pro_well,
        copy=True
    )

    # Copy the best solution files to a specific directory
    write_solution(
        locs, 
        keyword=key_dic['inj_rate_key'], 
        n_prod_well=n_pro_well, n_inj_well=n_optimum_well - n_pro_well,
        copy=True
    )


def write_gbf(main_path: str, gb_fitness: List[float]):
    """
    Write global best fitness values to a file.

    Args:
        main_path (str): Main path where the file should be created.
        gb_fitness (list of float): List of global best fitness values.
    """
    path = f'{main_path}/Global_best_Fitness'
    path_check(path) 

    with open(f'{path}/gbf.txt', 'w') as gbf_file:
        epoch = 1
        for item in gb_fitness:
            gbf_file.write(f'> Epoch {epoch}: GBF = {item}\n')
            epoch += 1


def read_best_rqi(file_name='RQI', mode='all'):
    """
    Read data from files containing best RQI values, locations, and null blocks.

    Args:
        file_name (str, optional): The directory containing the files. Default is 'RQI'.
        mode (str, optional): The mode for reading data ('all' or 'partial'). Default is 'all'.

    Returns:
        tuple: A tuple containing best RQI values, best RQI locations, and null block locations.
               The returned values depend on the mode parameter.
    """
    
    # Read best RQI values from file
    with open(f'{file_name}/best_RQI.text', 'r') as file:
        best_rqi = [float(i) for i in file]

    # Read best RQI locations from file
    with open(f'{file_name}/best_RQI_locs.text', 'r') as rqi_loc:
        best_rqi_locs = [tuple(map(int, i.split(','))) for i in rqi_loc]

    # Read null block locations from file
    with open(f'{file_name}/Null_blocks.text', 'r') as null_file:
        null_locs = [tuple(map(int, i.split(','))) for i in null_file]

    # Return different values based on the mode parameter
    if mode == 'all':
        return best_rqi, best_rqi_locs, null_locs

