import numpy as np
from benchmark import SimEnvironmentWBD, SimEnvironmentSwimmer, SimEnvironmentPVD, SimEnvironmentLunar, \
    SimEnvironmentAckley, SimEnvironmentStyblinski, SimEnvironmentTensionCompressionString, SimEnvironmentKeane, \
    SimEnvironmentSklearn, SimEnvironmentReacher, SimEnvironmentIllustration
from models import GPMultipleConstraints
from GPyOpt import Design_space
from constraint_acquisition import AcquisitionEICB, AcquisitionEIC, AcquisitionMESC, AcquisitionConstrainedThompson, AcquisitionEICB_CDF, AcquisitionPICB_CDF
import GPyOpt
from utils import Sequential, SobolDesign
import time
from typing import NamedTuple


class ConfigureStructure(NamedTuple):
    name: str
    is_partial: bool
    is_classification: bool
    obj_partial: bool
    feasible_value: float
    infeasible_value: float
    bo_algorithm: str
    iteration_num: int
    test: bool


def run_bo_experiment(seed: int, config: ConfigureStructure, hpoconfig=None, given_points=None):
    np.random.seed(seed)
    save_name = config.name
    classification = config.is_classification
    cons_partial = config.is_partial
    obj_partial = config.obj_partial
    if classification:
        feasible_value = config.feasible_value
        infeasible_value = config.infeasible_value
        if infeasible_value is None or feasible_value is None:
            raise ValueError()

    if config.name == 'en_wbd':
        sampling_num = 20
        constraint_num = 5
        space = Design_space(space=[
            {'name': 'var_1', 'type': 'continuous', 'domain': (0.125, 5)},
            {'name': 'var_2', 'type': 'continuous', 'domain': (0.1, 10)},
            {'name': 'var_3', 'type': 'continuous', 'domain': (0.1, 10)},
            {'name': 'var_4', 'type': 'continuous', 'domain': (0.1, 5)},
        ])
        envs = SimEnvironmentWBD(cons_partial=cons_partial, obj_partial=obj_partial, classification=classification,
                                 feasible_value=1.,
                                 infeasible_value=0.)
    elif config.name == 'illustration':
        sampling_num = 10
        constraint_num = 1
        space = Design_space(space=[
            {'name': 'var_1', 'type': 'continuous', 'domain': (0., 10.)},
        ])
        envs = SimEnvironmentIllustration(cons_partial=cons_partial, obj_partial=obj_partial, classification=classification,
                                 feasible_value=1.,
                                 infeasible_value=0.)
    elif config.name == 'en_pvd':
        sampling_num = 20
        constraint_num = 4
        space = Design_space(space=[
            {'name': 'var_1', 'type': 'continuous', 'domain': (0, 20)},
            {'name': 'var_2', 'type': 'continuous', 'domain': (0, 20)},
            {'name': 'var_3', 'type': 'continuous', 'domain': (10, 50)},
            {'name': 'var_4', 'type': 'continuous', 'domain': (150, 200)},
        ])
        envs = SimEnvironmentPVD(cons_partial=cons_partial, obj_partial=obj_partial, classification=classification,
                                 feasible_value=1.,
                                 infeasible_value=0.)
    elif config.name == 'en_tcs':
        sampling_num = 32
        constraint_num = 4
        given_points = np.array([[0.0971611, 1.07721786, 8.58139038]])
        space = Design_space(space=[
            {'name': 'var_1', 'type': 'continuous', 'domain': (0.05, 2)},
            {'name': 'var_2', 'type': 'continuous', 'domain': (0.25, 1.3)},
            {'name': 'var_3', 'type': 'continuous', 'domain': (5, 15)},
        ])
        envs = SimEnvironmentTensionCompressionString(cons_partial=cons_partial, obj_partial=obj_partial,
                                                      classification=classification, feasible_value=1.,
                                                      infeasible_value=0.)
    elif config.name == 'rl_swimmer':
        sampling_num = 16 * 11
        constraint_num = 1
        space = Design_space(space=[
            {'name': 'var_1', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_2', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_3', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_4', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_5', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_6', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_7', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_8', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_9', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_10', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_11', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_12', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_13', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_14', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_15', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_16', 'type': 'continuous', 'domain': (-1, 1)},
        ])
        envs = SimEnvironmentSwimmer(cons_partial=cons_partial, classification=classification, feasible_value=1.,
                                     infeasible_value=0.)
    elif config.name == 'rl_lunar':
        sampling_num = 12 * 11
        constraint_num = 1
        space = Design_space(space=[
            {'name': 'var_1', 'type': 'continuous', 'domain': (0., 2.)},
            {'name': 'var_2', 'type': 'continuous', 'domain': (0., 2.)},
            {'name': 'var_3', 'type': 'continuous', 'domain': (0., 2.)},
            {'name': 'var_4', 'type': 'continuous', 'domain': (0., 2.)},
            {'name': 'var_5', 'type': 'continuous', 'domain': (0., 2.)},
            {'name': 'var_6', 'type': 'continuous', 'domain': (0., 2.)},
            {'name': 'var_7', 'type': 'continuous', 'domain': (0., 2.)},
            {'name': 'var_8', 'type': 'continuous', 'domain': (0., 2.)},
            {'name': 'var_9', 'type': 'continuous', 'domain': (0., 2.)},
            {'name': 'var_10', 'type': 'continuous', 'domain': (0., 2.)},
            {'name': 'var_11', 'type': 'continuous', 'domain': (0., 2.)},
            {'name': 'var_12', 'type': 'continuous', 'domain': (0., 2.)},
        ])
        envs = SimEnvironmentLunar(cons_partial=cons_partial, classification=classification, feasible_value=1.,
                                   infeasible_value=0.)

    elif config.name == 'rl_reacher':
        sampling_num = 22 * 11
        constraint_num = 1
        space = Design_space(space=[
            {'name': 'var_1', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_2', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_3', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_4', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_5', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_6', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_7', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_8', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_9', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_10', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_11', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_12', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_13', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_14', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_15', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_16', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_17', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_18', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_19', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_20', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_21', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'var_22', 'type': 'continuous', 'domain': (-1, 1)},
        ])
        envs = SimEnvironmentReacher(cons_partial=cons_partial, classification=classification, feasible_value=1.,
                                   infeasible_value=0.)

    elif config.name == 'sy_ackley':
        sampling_num = 10 * 1
        constraint_num = 1
        space = Design_space(space=[
            {'name': 'var_1', 'type': 'continuous', 'domain': (-5., 5.)},
            {'name': 'var_2', 'type': 'continuous', 'domain': (-5., 5.)},
            {'name': 'var_3', 'type': 'continuous', 'domain': (-5., 5.)},
            {'name': 'var_4', 'type': 'continuous', 'domain': (-5., 5.)},
            {'name': 'var_5', 'type': 'continuous', 'domain': (-5., 5.)},
            {'name': 'var_6', 'type': 'continuous', 'domain': (-5., 5.)},
            {'name': 'var_7', 'type': 'continuous', 'domain': (-5., 5.)},
            {'name': 'var_8', 'type': 'continuous', 'domain': (-5., 5.)},
            {'name': 'var_9', 'type': 'continuous', 'domain': (-5., 5.)},
            {'name': 'var_10', 'type': 'continuous', 'domain': (-5., 5.)},
        ])
        envs = SimEnvironmentAckley(cons_partial=cons_partial, obj_partial=obj_partial, classification=classification,
                                    feasible_value=1.,
                                    infeasible_value=0.)

    elif config.name == 'sy_styblinski':
        sampling_num = 8 * 11
        constraint_num = 1
        space = Design_space(space=[
            {'name': 'var_1', 'type': 'continuous', 'domain': (-5., 5.)},
            {'name': 'var_2', 'type': 'continuous', 'domain': (-5., 5.)},
            {'name': 'var_3', 'type': 'continuous', 'domain': (-5., 5.)},
            {'name': 'var_4', 'type': 'continuous', 'domain': (-5., 5.)},
            {'name': 'var_5', 'type': 'continuous', 'domain': (-5., 5.)},
            {'name': 'var_6', 'type': 'continuous', 'domain': (-5., 5.)},
            {'name': 'var_7', 'type': 'continuous', 'domain': (-5., 5.)},
            {'name': 'var_8', 'type': 'continuous', 'domain': (-5., 5.)},
        ])
        envs = SimEnvironmentStyblinski(cons_partial=cons_partial, obj_partial=obj_partial,
                                        classification=classification, feasible_value=1.,
                                        infeasible_value=0.)
    elif config.name == 'sy_kbf5':
        sampling_num = 5 * 11
        constraint_num = 2
        space = Design_space(space=[
            {'name': 'var_1', 'type': 'continuous', 'domain': (0, 10)},
            {'name': 'var_2', 'type': 'continuous', 'domain': (0, 10)},
            {'name': 'var_3', 'type': 'continuous', 'domain': (0, 10)},
            {'name': 'var_4', 'type': 'continuous', 'domain': (0, 10)},
            {'name': 'var_5', 'type': 'continuous', 'domain': (0, 10)},
        ])
        envs = SimEnvironmentKeane(cons_partial=cons_partial, obj_partial=obj_partial,
                                   classification=classification, feasible_value=1.,
                                   infeasible_value=0.)
    elif config.name == 'sy_kbf10':
        sampling_num = 10 * 11
        constraint_num = 2
        space = Design_space(space=[
            {'name': 'var_1', 'type': 'continuous', 'domain': (0, 10)},
            {'name': 'var_2', 'type': 'continuous', 'domain': (0, 10)},
            {'name': 'var_3', 'type': 'continuous', 'domain': (0, 10)},
            {'name': 'var_4', 'type': 'continuous', 'domain': (0, 10)},
            {'name': 'var_5', 'type': 'continuous', 'domain': (0, 10)},
            {'name': 'var_6', 'type': 'continuous', 'domain': (0, 10)},
            {'name': 'var_7', 'type': 'continuous', 'domain': (0, 10)},
            {'name': 'var_8', 'type': 'continuous', 'domain': (0, 10)},
            {'name': 'var_9', 'type': 'continuous', 'domain': (0, 10)},
            {'name': 'var_10', 'type': 'continuous', 'domain': (0, 10)},
        ])
        envs = SimEnvironmentKeane(cons_partial=cons_partial, obj_partial=obj_partial,
                                   classification=classification, feasible_value=1.,
                                   infeasible_value=0.)
    elif config.name == 'skl':
        assert hpoconfig is not None
        save_name = save_name + '_' + hpoconfig['dataset'] + '_' + hpoconfig['method'] + '_' + str(
            hpoconfig['threshold_ratio'])
        if hpoconfig['method'] == 'rf':
            sampling_num = 5 * 11
            space = Design_space(space=[
                {'name': 'var_1', 'type': 'continuous', 'domain': (np.log10(1), np.log10(50))},
                {'name': 'var_2', 'type': 'continuous', 'domain': (np.log2(2), np.log2(128))},
                {'name': 'var_3', 'type': 'continuous', 'domain': (np.log10(1), np.log10(100))},
                {'name': 'var_4', 'type': 'continuous', 'domain': (1, 21 - 1e-10)},
                {'name': 'var_5', 'type': 'continuous', 'domain': (0. + 1e-6, 1. - 1e-6)},
            ])
        elif hpoconfig['method'] == 'xgb':
            sampling_num = 7 * 11
            space = Design_space(space=[
                {'name': 'var_1', 'type': 'continuous', 'domain': (-10, 0)},
                {'name': 'var_2', 'type': 'continuous', 'domain': (1, 15)},
                {'name': 'var_3', 'type': 'continuous', 'domain': (0.01, 1.)},
                {'name': 'var_4', 'type': 'continuous', 'domain': (-10, 10)},
                {'name': 'var_5', 'type': 'continuous', 'domain': (-10, 10)},
                {'name': 'var_6', 'type': 'continuous', 'domain': (0, 7)},
                {'name': 'var_7', 'type': 'continuous', 'domain': (0, 8)},
            ])
        elif hpoconfig['method'] == 'nn':
            sampling_num = 8 * 11
            space = Design_space(space=[
                {'name': 'var_1', 'type': 'continuous', 'domain': (2, 8)},
                {'name': 'var_2', 'type': 'continuous', 'domain': (2, 8)},
                {'name': 'var_3', 'type': 'continuous', 'domain': (2, 8)},
                {'name': 'var_4', 'type': 'continuous', 'domain': (-8, -3)},
                {'name': 'var_5', 'type': 'continuous', 'domain': (-5, 0)},
                {'name': 'var_6', 'type': 'continuous', 'domain': (-6, -2)},
                {'name': 'var_7', 'type': 'continuous', 'domain': (0, 0.9999)},
                {'name': 'var_8', 'type': 'continuous', 'domain': (0, 0.9999)},
            ])
        else:
            print("No method of HPO experiment determined!")
            raise ValueError()
        constraint_num = 1

        envs = SimEnvironmentSklearn(cons_partial=cons_partial,
                                     classification=classification, feasible_value=1.,
                                     infeasible_value=0., method=hpoconfig['method'], dataset=hpoconfig['dataset'],
                                     memory_ratio=hpoconfig['threshold_ratio'])

    else:
        print("No experiment determined!")
        raise ValueError()

    sampler = SobolDesign(space)
    sample_points = sampler.get_samples(sampling_num)

    # '''
    #     test to init from data
    # '''
    # dir = './init_data/swimmer/swimmer-seed' + str(seed) + '.txt'
    # sample_points = np.loadtxt(dir)

    if given_points is not None:
        sample_points = np.vstack((sample_points, given_points))

    observations = envs.observer(sample_points)
    CBO_model = GPMultipleConstraints(dataset=observations, constraint_num=constraint_num,
                                      classification=classification)

    iteration_num = config.iteration_num

    acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space)
    if config.bo_algorithm == 'eic':
        acquisition = AcquisitionEIC(CBO_model, space, acquisition_optimizer, classification=classification)
        evaluator = Sequential(acquisition, is_sampling=False)
    elif config.bo_algorithm == 'eicb':
        acquisition = AcquisitionEICB(CBO_model, space, acquisition_optimizer, classification=classification)
        evaluator = Sequential(acquisition, updating=True, is_sampling=False)
    elif config.bo_algorithm == 'eicb_cdf':
        acquisition = AcquisitionEICB_CDF(CBO_model, space, acquisition_optimizer, classification=classification, beta=2.)
        evaluator = Sequential(acquisition, updating=False, is_sampling=False)
    elif config.bo_algorithm == 'picb_cdf':
        acquisition = AcquisitionPICB_CDF(CBO_model, space, acquisition_optimizer, classification=classification, beta=2.)
        evaluator = Sequential(acquisition, updating=False, is_sampling=False)
    elif config.bo_algorithm == 'mesc':
        acquisition = AcquisitionMESC(CBO_model, space, acquisition_optimizer)
        evaluator = Sequential(acquisition, is_sampling=True)
    elif config.bo_algorithm == 'thompsonc':
        acquisition = AcquisitionConstrainedThompson(CBO_model, space, acquisition_optimizer,
                                                     classification=classification)
        evaluator = Sequential(acquisition, is_sampling=True)
    else:
        print("No algorithm specified.")
        raise ValueError()

    best_value = np.zeros((iteration_num + 1, 1))
    best_value[0] = CBO_model.obj_min()
    for i in range(iteration_num):

        # calculate the argmax of acquisition functions
        print("##### calculate the argmax of acquisition functions #####")
        time_acf_start = time.time()
        suggested_sample, acq_value = evaluator.next_point(None, context_manager=None)
        time_acf_end = time.time()
        print("[iter{}] AcFun:\t\t{:.0f}h{:.0f}m{:.1f}s".format(
            i,
            (time_acf_end - time_acf_start) / 3600,
            (time_acf_end - time_acf_start) % 3600 / 60,
            (time_acf_end - time_acf_start) % 3600 % 60, ))

        print("##### evaluating the objective and constraint functions #####")
        # evaluate the objective and constraint values
        new_observation = envs.observer(suggested_sample)

        # update the dataset and model
        print("##### updating the surrogate models #####")
        time_fit_start = time.time()
        try:
            CBO_model.update(new_observation)
        except Exception as e:
            print("# # # # # # During model updating  something goes wrong # # # # # # ")
            print("# # # # # #See the following statements for more details# # # # # # ")
            print(e.args)
            exit()
        time_fit_end = time.time()

        print("[iter{}] FitModel:\t{:.0f}h{:.0f}m{:.1f}s".format(
            i,
            (time_fit_end - time_fit_start) / 3600,
            (time_fit_end - time_fit_start) % 3600 / 60,
            (time_fit_end - time_fit_start) % 3600 % 60, ))
        best_value[i + 1] = min(CBO_model.obj_min(), best_value[i])

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(np.linspace(1, iteration_num + 1, iteration_num + 1), best_value)
    # plt.show()

    if classification:
        dir_loc = save_name + '/' + config.bo_algorithm + '_classification/' + config.bo_algorithm + '_classification_' + str(
            seed)
    elif cons_partial:
        dir_loc = save_name + '/' + config.bo_algorithm + '_partial/' + config.bo_algorithm + '_partial_' + str(seed)
    else:
        dir_loc = save_name + '/' + config.bo_algorithm + '/' + config.bo_algorithm + '_' + str(seed)

    obj_data = np.hstack((CBO_model.dataset[0][0], CBO_model.dataset[0][1]))
    con_data = np.hstack((CBO_model.dataset[1][0][0], CBO_model.dataset[1][0][0]))

    if config.test:
        np.savetxt('./test_results/' + dir_loc + '_data_obj.txt', obj_data)
        np.savetxt('./test_results/' + dir_loc + '_data_con.txt', con_data)
    else:
        np.savetxt('./results/' + dir_loc + '_data_obj.txt', obj_data)
        np.savetxt('./results/' + dir_loc + '_data_con.txt', con_data)


if __name__ == '__main__':
    hpo_config = {
        'method': 'xgb',
        'dataset': 'steel',
        'threshold_ratio': 0.5
    }
    config = ConfigureStructure(
        name='illustration',
        is_partial=False,
        obj_partial=True,
        is_classification=False,
        feasible_value=None,
        infeasible_value=None,
        bo_algorithm='picb_cdf',
        iteration_num=10,
        test=True,
    )
    # run_bo_experiment(seed=2, config=config, hpoconfig=hpo_config)
    for i in range(10):
        run_bo_experiment(seed=i + 1, config=config)
