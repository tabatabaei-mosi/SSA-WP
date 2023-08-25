import time
from typing import Dict, List, Tuple, Union

from loguru import logger
from mealpy.swarm_based import SSA

from constrains import loc_penalty, null_block_penalty
from optimizer import SSA_WP
from utils import (batch_summary, final_result, get_simulation_info,
                   path_check, read_best_rqi, ssa_log, write_gbf)


def main(
    optimization_keys: Dict,
    prop_key: List[str],
    n_well_prod: int = 30,
    n_well_inj: int = 5,
    capex_value: int = 175000000,
    threshold: int = 0,
    well_space: int = 2,
    null_space: int = 2,
    grid_dim: Tuple[int, int, int] = (85, 185, 31),
    unit_npv: str = "M",
    log_path: str = "Log_Files",
    grid_path: str = "INCLUDE/SSA_RQI_0.GRDECL",
    optimizer_hyperparams: Dict = {
        'iteration': 50, 'pop_size': 20,
        'ST': 0.85, 'PD': 0.72, 'SD': 0.08
    },
):
    """
    Execute the optimization process for well placement using SSA algorithm.

    Arguments:
        optimization_keys {Dict} -- Dictionary containing keys for well optimization. For example:

        {
            'loc': 'WELSPECS',
            'perf': 'COMPDAT',
            'pro_rate': 'WCONPROD',
            'inj_rate': 'WCONINJE'
        }

        prop_key {List[str]} -- List of property keywords.

    Keyword Arguments:
        n_well_prod (int, optional): Number of production wells. Default is 30.
        n_well_inj (int, optional): Number of injection wells. Default is 5.
        capex_value (int, optional): Capital expenditure value. Default is 175000000.
        threshold (int, optional): RQI threshold value. Default is 0.
        well_space (int, optional): Minimum well spacing. Default is 2.
        null_space (int, optional): Minimum distance to null blocks. Default is 2.
        grid_dim (Tuple[int, int, int], optional): Grid dimensions. Default is (85, 185, 31).
        unit_npv (str, optional): NPV unit. Default is "M".
        log_path (str, optional): Path to save log files. Default is "Log_Files".
        grid_path (str, optional): Path to the grid file. Default is "INCLUDE/SSA_RQI_0.GRDECL".
        optimizer_hyperparams (Dict, optional): Dictionary containing optimizer hyperparameters.
    """
    # Start the timer to calculate the total run time.
    start = time.time()

    # Number of wells to be optimized.
    t_opt_well = n_well_prod + n_well_inj

    # Create a Log_Files directory to save all the ouputs there.
    path_check(log_path)

    # Clean the map and generate best RQI map based on the threshold.
    best_rqi, best_rqi_locs, null_blocks = read_best_rqi(
        file_name=f"RQI/TH={threshold}/Interval_checkup",
        threshold=threshold,
        null_space=null_space,
        prop_key=prop_key,
        grid_path=grid_path,
        dim=grid_dim,
        mode="all"
    )

    # Create an object of the SSA_WP class (Coupled SSA and WP Problem)
    ssa_wp_obj = SSA_WP(
        prod_well=n_well_prod,
        inj_well=n_well_inj,
        well_space=well_space,
        keys=optimization_keys,
        dims=grid_dim,
        best_rqi_locs=best_rqi_locs,
        capex_value=capex_value,
        log_path=log_path
    )

    # Define the problem feed to optimizer:
    problem_dict = {
        "fit_func": ssa_wp_obj.obj_func,
        "lb": [0] * t_opt_well + [1000] * n_well_prod + [1000] * n_well_inj,
        "ub": [len(best_rqi) - 1] * t_opt_well + [10000] * n_well_prod + [10000] * n_well_inj,
        "minmax": "max",
        "amend_position": ssa_wp_obj.amend_position,
        "log_to": "file",
        "log_file": "Log_Files/SSA.log"
    }

    # Define the termination criteria and other hyperparameters of optimizer.
    model = SSA.BaseSSA(
        problem_dict,
        epoch=optimizer_hyperparams['iteration'],
        pop_size=optimizer_hyperparams['pop_size'],
        PD=optimizer_hyperparams['PD'],
        SD=optimizer_hyperparams['SD'],
        ST=optimizer_hyperparams['ST']
    )

    # Run the optimizer.
    best_positions, best_fitness = model.solve()

    # Save the results from model.history
    model.history.save_global_best_fitness_chart(
        filename=f"{log_path}/Charts/gbfc",
        y_label=f'NPV ($ {unit_npv})'
    )

    model.history.save_exploration_exploitation_chart(
        filename=f"{log_path}/Charts/eec"
    )
    gb_solution = model.history.list_global_best
    gbf_ = model.history.list_global_best_fit

    # Save the best NPV
    write_gbf(log_path, gbf_)

    # End of total run time
    end = time.time()
    run_time = abs(start - end)

    # Put essential info about run of optimizer into a list
    ssa_output = [
        run_time,
        best_positions, best_fitness,
        optimizer_hyperparams['PD'],
        optimizer_hyperparams['SD'],
        optimizer_hyperparams['ST'],
        optimizer_hyperparams['iteration'],
        optimizer_hyperparams['pop_size']
    ]
    # Finilize the SSA Log by adding info about hyper-parameters, GBS, GBF, and runtime
    # Save all details in mealpy log file (Verbose=True)
    ssa_log(filename='SSA.log', solution=ssa_output, npv_unit=unit_npv)

    # Summerize the batch file results (summary of all simulation call) ...
    # ... For example, how many Errors occured, in which call? etc.
    batch_summary()

    # Extract meaningful information about [a list of keys] from summary file
    # List of keys when no args are passed is: ['PROBLEM', 'WARNING', 'ERROR']
    get_simulation_info()

    # Check the Quality of GBS and find whether violate any constains or not
    # 1. Check well space
    loc_space_check, _ = loc_penalty(
        solution=best_positions,
        target='min_space',
        n_optwell=t_opt_well,
        min_well_space=well_space,
        grid_size=grid_dim,
        best_rqi_locs=best_rqi_locs,
        obj_func=False
    )

    # 2. Check borders
    loc_border, _ = loc_penalty(
        solution=best_positions,
        target='border',
        n_optwell=t_opt_well,
        min_well_space=well_space,
        grid_size=grid_dim,
        best_rqi_locs=best_rqi_locs,
        obj_func=False
    )

    # 3. min space to null blocks
    loc_null, _ = null_block_penalty(
        opt_solution=best_positions,
        null_block_list=null_blocks,
        best_rqi_locs=best_rqi_locs,
        opt_well_number=t_opt_well,
        min_null_space=null_space,
        obj_func=False
    )

    # Create a file with final output evaluation and records
    final_result(
        Optimizer_info=ssa_output,
        capex_value=capex_value,
        npv_unit=unit_npv,
        n_pro_well=n_well_prod,
        n_optimum_well=t_opt_well,
        well_space=well_space,
        key_dic=optimization_keys,
        best_rqi_locs=best_rqi_locs,
        dims=grid_dim,
        spaces=loc_space_check,
        border=loc_border,
        null=loc_null
    )

    logger.info(f'Process is finished. {round(run_time / 60, 3)} min.')


if __name__ == "__main__":

    # Define which keywords in .DATA file are used for optimization
    optimization_keys = {
        'loc': 'WELSPECS',
        'perf': 'COMPDAT',
        'pro_rate': 'WCONPROD',
        'inj_rate': 'WCONINJE'
    }

    # Constrains for optimization
    # 1. Minimum well spacing
    # 2. Minimum distance to null blocks
    well_space, null_space = 2, 2

    # Define the properties used for calculating the RQI (Read from .GRDECL file)
    prop_key = ['PORO', 'PERMX', 'PERMY', 'PERMZ']

    # Path to the grid file
    grid_path = "INCLUDE/SSA_RQI_0.GRDECL"

    #
    reservoir_dim = (85, 185, 31)

    # Threshold value for RQI
    threshold = 0

    # number of production and injection wells
    n_well_prod, n_well_inj = 30, 5

    # Capex value
    capex_value = (n_well_prod + n_well_inj) * 5000000
    npv_unit = "M"

    # Where to save the log files
    log_path = "Log_Files"

    # Optimizer hyperparameters
    optimizer_hyperparams = {
        'iteration': 50, 'pop_size': 20,
        'ST': 0.85, 'PD': 0.72, 'SD': 0.08
    }

    main(
        optimization_keys=optimization_keys,
        prop_key=prop_key,
        n_well_prod=n_well_prod,
        n_well_inj=n_well_inj,
        capex_value=capex_value,
        threshold=threshold,
        well_space=well_space,
        null_space=null_space,
        grid_dim=reservoir_dim,
        unit_npv=npv_unit,
        log_path=log_path,
        grid_path=grid_path,
        optimizer_hyperparams=optimizer_hyperparams
    )
