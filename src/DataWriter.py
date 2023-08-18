from pathlib import Path
from typing import List, Union
from utils import path_check

import numpy as np
from loguru import logger

class LogProcessor:
    def __init__(self):
        pass
    
    def solution_track(
        self,
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
        self,
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
        self,
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
        self,
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
        self,
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
        self,
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
            print(f"An error occurred while writing to the file: {e}")


    def write_solution(
        self,
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
            content = self.prepare_welspecs_content(solution, n_prod_well, n_inj_well)

        elif keyword == 'COMPDAT':
            content = self.prepare_compdat_content(solution, n_prod_well, n_inj_well)

        elif keyword == 'WCONPROD':
            content = self.prepare_wconprod_content(solution, n_prod_well)

        elif keyword == 'WCONINJE':
            content = self.prepare_wconinje_content(solution, n_inj_well, track=track)

        logger.debug(f"Writing content to file: {file_path}")
        self.write_to_file(file_path, content)
        
        if track:
            track_file_name = f'track_{file_name}.txt'
            self.solution_track(content, filename=track_file_name)