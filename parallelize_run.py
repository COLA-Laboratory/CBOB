import multiprocessing
from single_run import ConfigureStructure, run_bo_experiment

if __name__ == '__main__':
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ]  # List of initial seeds
    # seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # List of initial seeds
    # seeds = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    hpo_config = {
        'method': 'xgb',
        'dataset': 'house',
        'threshold_ratio': 0.5
    }
    config = ConfigureStructure(
        name='sy_kbf10',
        is_partial=False,
        obj_partial=True,
        is_classification=False,
        feasible_value=None,
        infeasible_value=None,
        bo_algorithm='eicb_cdf',
        iteration_num=200,
        test=False,
    )
    
    # Create a pool of worker processes
    pool = multiprocessing.Pool()
    
    # Run the code with different seeds in parallel, passing extra arguments
    pool.starmap(run_bo_experiment, [(seed, config, hpo_config) for seed in seeds])
    
    # Close the pool of worker processes
    pool.close()
    pool.join()
