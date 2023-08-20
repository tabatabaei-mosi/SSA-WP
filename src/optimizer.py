import numpy as np
import pandas as pd
import subprocess
from typing import List, Tuple
from utils import final_idx_check, write_solution
from constrains import logical_constrains

class SSA_WP:
    def __init__(
        self, 
        prod_well: int, inj_well: int, 
        well_space: int, 
        keys: dict,
        dims: Tuple[int, int, int],
        best_rqi_locs,
        capex_value: float, 
        log_path: str = 'Log_Files'
    ):
        """
        Initialize an instance of the SSA_WP class. This class will serve as 

        Args:
            prod_well (int): The number of production wells.
            inj_well (int): The number of injection wells.
            well_space (int): The minimum distance between wells.
            keys (dict): A dictionary containing the keywords for different section of .DATA file.
                         Expected keys: 'loc', 'perf', 'pro_rate', 'inj_rate'
            dims (tuple): The model dimention (grid blocks)
            best_rqi_locs (list): A list of promising solutions: [(0, 0, 1), (2, 5, 1), ...]
            capex_value (float): The capital expenditure value for NPV calculations.
            log_path (str): The path for logging outputs.
        """
        # Initialize class attributes
        self.prod_well = prod_well
        self.inj_well = inj_well
        self.t_opt_well = inj_well + prod_well
        self.well_space = well_space
        self.loc_key = keys['loc']
        self.perf_key = keys['perf']
        self.pro_rate_key = keys['pro_rate']
        self.inj_rate_key = keys['inj_rate']
        self.dims = dims
        self.best_rqi_locs = best_rqi_locs
        self.capex_value = capex_value
        self.log_path = log_path


    @staticmethod
    def amend_position(
        solution: np.ndarray, 
        lower_bound: float, upper_bound: float
    ) -> np.ndarray:
        """Amend solution positions to fit within bounds."""
        pos = np.clip(solution, lower_bound, upper_bound)
        return pos


    def split_solution(self, solution: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split solution into location IDs, production rates, and injection rates.
        
        Args:
            solution: The row format of solution generated by optimizer. Because we're optimizing location and flow rate
            of production and injection wells, so we assume that the variable are in this order:
            [prod locs, inj locs, prod rates, inj rates], it means that if we have 30 production and 6 injection wells;
            therefore, 30 first variable are locations of production wells, followed by 6 location of injection wells, and so on.
            
            NOTE: We get a float number from optimizer, map this to a list of locations and will find which location it means.
            For example: optimizer will generate, 3.587 and this will transform to 3 and later map to (7, 2, 1) loc.
        
        """
        # Split the locations from solution (the values are float)
        loc_id_float = solution[0:self.t_opt_well]
        # Convert float to int
        loc_id_int = loc_id_float.astype(int)
        
        # Split the rates from solution (the values are float)
        well_rates_floats = solution[self.t_opt_well:]
        # Convert float to int
        well_rates = well_rates_floats.astype(int)
        
        # Split the rates into production and injection rates
        prod_rates = well_rates[:self.prod_well]
        inj_rates = well_rates[self.prod_well:]
        
        return loc_id_int, prod_rates, inj_rates


    def map_backward(
        self,
        locs_id: np.ndarray, 
    ) -> List[Tuple[int, int, int]]:
        """
        Map location IDs to well locations using best RQI locations.
        NOTE: We get a float number from optimizer, make it integer and map this to a tuple of location from a list of best possible locations
        according to Rock Quality Index (RQI).
        
        """
        locs_list = []
        
        # Map location IDs to well locations
        for idx in locs_id:
            # Check if the index is within the range of best_rqi_loc_
            final_idx = final_idx_check(idx, len_obj_map=len(self.best_rqi_locs))
            
            # Get the well location from best_rqi_loc_
            well_loc = self.best_rqi_locs[final_idx]
            
            locs_list.append(well_loc)
            
        return locs_list


    def decode_locs(
        self,
        solution: List[Tuple[float, float, float]]
    ) -> Tuple[List[float], List[float]]:
        """
        Decode well locations and perforations. It means that locations and perforation both should start from 1 and
        we assume that perforation will extend 2 layers above and below the perforation layer chosen by optimizer.
        In fact, optimizer will generate a float number (e.g., 3.589) and this will transform to 3 and later map to tuple
        indicating a location within quality map (e.g., (7, 2, 1) loc).
        This suggested that perforation be done in layer 2 and then we will extend it to layer 1 and 3 (assumption).
        
        Args:
            solution: The splited, mapped solution.
        
        """
        locs = []
        perfs = []
        
        for well_locs in solution:
            # Add 1 to each location to make it start from 1 (.DATA works with 1-based indexing)
            loc_i = well_locs[0] + 1
            loc_j = well_locs[1] + 1
            locs.append(loc_i)
            locs.append(loc_j)
            
            # Add 1 to each perforation layer to make it start from 1
            perf_layer = well_locs[2] + 1
            
            # Extend perforation layer to 2 layers above and below the chosen layer
            # if it started by first layer, extend to 2 below first layer
            if perf_layer == 1:
                perf_start = perf_layer
                perf_end = perf_start + 2
            
            # if it perforated at the last layer, extend it to 2 layer above.    
            elif perf_layer == self.dimens[-1]:
                perf_end = perf_layer
                perf_start = perf_end - 2
            
            # Otherwise, one layer up and one layer down extended.
            else:
                perf_start = perf_layer - 1
                perf_end = perf_layer + 1
            
            perfs.append(perf_start)
            perfs.append(perf_end)
            
        return locs, perfs

    
    def npv_calculator(self, capex: float = 0) -> float:
        """
        Calculate Net Present Value (NPV) using provided parameters.
        
        Args: 
            capex: The CAPEX value in NPV formula.
        """
        # read the RSM File (resulted from simulation of "RAN.DATA")
        result_df = pd.read_fwf('RUN.RSM', sep=" ", header=1)
        
        # Get the final index where table is finished
        final_index = result_df['FOPT'].last_valid_index()
        
        # Read the NPV constants (From Excel)
        price_info = pd.read_excel('NPV.xlsx', sheet_name='Info')
        price_info = price_info.set_index('Parameter')
        
        # Assign the prices to variables
        r_o = float(price_info['ro']['Price'])
        r_wi = float(price_info['rwi']['Price'])
        r_wp = float(price_info['rwp']['Price'])
        
        # If any of following column have a special scale (unit), record it
        col_names = ['FOPT', 'FWPT', 'FWIT']
        col_units = []
        
        for name in col_names:
            # For example if it's 10**3 means that all value should multiply by 10**3
            if '*' in str(result_df[name][1]):
                unit_str = result_df[name][1]
                unit_list = unit_str[1:].split("**")
                unit = int(unit_list[0]) ** int(unit_list[1])
                col_units.append(unit)
            else:
                col_units.append(1)

        # All the times tables will start by index 1 unless, a column have a unit!
        # Start value is necessory for loop thorugh table in below For loop
        start = 1
        for unit in col_units:
            if unit != 1:
                start = 2
            
        # counter is time consideration in NPV formula
        counter = 0
        npv = 0
        for i in range(start, final_index + 1):
            counter += 1
            FOPT_i = float(result_df[col_names[0]][i]) * col_units[0]
            FWPT_i = float(result_df[col_names[1]][i]) * col_units[1]
            FWIT_i = float(result_df[col_names[2]][i]) * col_units[2]
            
            npv = npv + (((FOPT_i * r_o) - (FWPT_i * r_wp) - (FWIT_i * r_wi)) / (1.01 ** counter))
            
        NPV = npv - capex
        
        return NPV

    def obj_func(self, solution: np.ndarray) -> float:
        """
        Objective function to be optimized.
        
        """
        # splice the solution to different optimization variables
        locs_id, prod_rate, inj_rate = self.split_solution(solution)
        
        # map the ids generated by optimizer to quality map (grid location)
        well_locs = self.map_backward(locs_id)
        
        # decode the well_locs to be appropriate for .DATA file
        locs, perfs = self.decode_locs(well_locs)
        
        # Check if two well is located at same place or not
        penalty_1 = logical_constrains(locs, self.t_opt_well, target='loc') 
        
        # Check how many well violated the min_space between wells
        _, space_faults = self.loc_penalty(locs, target='min_space', n_optwell=self.t_opt_well, min_well_space=self.well_space,
                                  obj_func=True)
        # If a logical constrain violated, NPV = 0; otherwise, run the simulation
        # .... and the punish the algorithm (reduce the NPV) based on the penalty coeff and the number of violation 
        if penalty_1 == 1:
            return 0
        else:
            # write the decoded solution (locations) of optimizer 
            write_solution(
                locs, 
                keyword=self.loc_key, 
                n_prod_well=self.prod_well, n_inj_well=self.inj_well,
                track=True
            )
            
            # write the decoded solution (perforations) of optimizer 
            write_solution(
                perfs, 
                keyword=self.perf_key, 
                n_prod_well=self.prod_well, n_inj_well=self.inj_well,
                track=True
            )
            
            # write the decoded solution (production rates) of optimizer 
            write_solution(
                prod_rate, 
                keyword=self.pro_rate_key, 
                n_prod_well=self.prod_well, n_inj_well=self.inj_well,
                track=True
            )

            # write the decoded solution (inj rates) of optimizer 
            write_solution(
                inj_rate, 
                keyword=self.inj_rate_key, 
                n_prod_well=self.prod_well, n_inj_well=self.inj_well,
                track=True
            )

            # Call the ECLIPSE to run the simulation, write out the results of simulation in a file
            with open(f'{self.log_path}/bat_results.txt', 'a') as batch_outputs:
                subprocess.call([r"$MatEcl.bat"], stdout=batch_outputs)
            
            # Now Calculate the NPV based on the last simulation run 
            npv_value = self.npv_calculator(capex=self.capex_value)
            
            # If min_space is violated, for each reduced the NPV by 30%. If it excess the 1, then NPV=0
            if space_faults >= 1:
                dis_punishment = 0.3 * space_faults
                total_punish = dis_punishment
                
                if total_punish > 1:
                    return 0
                
                else:
                    npv_value = npv_value - (total_punish * npv_value)
                    return npv_value / (10 ** 9)
                
            else:
                return npv_value / (10 ** 9)
