import numpy as np
from scipy.stats import norm
import warnings


def ep_truncated(y, mean, K, iteration_num, a=0.0, partial=False):
    """
    This is an implementation of EP method using truncated function, see R. Garnett, Bayesian Optimization, Cambridge University Press 2023.
    y: observations
    mean: latent mean function
    K: covariance matrix
    iteration_num: iteration number of EP alg.
    a: threshold of truncated distributions, default 0 for partially observable constraints
    partial: True refers to partially observable constraints, False refers to unobservable constraints, i.e. binary constraints (GPC)
    """
    log_2_pi = np.log(2 * np.pi)
    # initialize the parameters
    dim = K.shape[0]
    nu_til_vec = np.zeros(dim)
    tau_til_vec = np.zeros(dim)
    mu_vec = mean * 0.
    Sigma = np.copy(K)

    mu_cav_vec = np.zeros(dim)
    var_cav_vec = np.zeros(dim)
    tau_cav_vec = np.zeros(dim)
    nu_cav_vec = np.zeros(dim)
    logZ_til_vec = np.zeros(dim)
    logZ_hat_vec = np.zeros(dim)

    norm_distribution = norm(loc=0., scale=1.)
    iteration = 0
    while iteration < iteration_num:
        for i in range(dim):

            # compute the approximate cavity parameters
            tau_cav_i = 1. / Sigma[i, i] - tau_til_vec[i]
            nu_cav_i = mu_vec[i] / Sigma[i, i] - nu_til_vec[i]
            if tau_cav_i < 1e-3:
                raise ValueError("tau_cav_i is too small! (please change it to a stable version)")
            tau_cav_vec[i] = tau_cav_i
            nu_cav_vec[i] = nu_cav_i

            # compute the marginal moments (only implement the truncated distribution)
            var_cav_i = 1. / tau_cav_i
            mu_cav_i = var_cav_i * nu_cav_i
            var_cav_vec[i] = var_cav_i
            mu_cav_vec[i] = mu_cav_i

            if not partial or y[i] > 0:
                std_cav_i = np.sqrt(var_cav_i)
                z_i = (a - mu_cav_i) / std_cav_i
                pdf = norm_distribution.pdf(z_i)
                if y[i] > 0:
                    cdf = 1 - norm_distribution.cdf(z_i)
                    if cdf < 1e-6:
                        cdf = 1e-6
                    alpha = pdf / (cdf * std_cav_i)
                else:
                    cdf = norm_distribution.cdf(z_i)
                    alpha = - pdf / (cdf * std_cav_i)
                beta = 0.5 * z_i * alpha / std_cav_i
                if y[i] > 0:
                    gamma = 1 / (alpha**2 - 2 * beta)
                else:
                    gamma = - std_cav_i / alpha / (pdf / cdf + z_i)

                var_til_new_i = gamma - var_cav_i
                mu_til_new_i = mu_cav_i + alpha * gamma
                Z = cdf
                Delta_tau_til = tau_til_vec[i]
                tau_til_vec[i] = 1. / var_til_new_i
                nu_til_vec[i] = mu_til_new_i / var_til_new_i
                Delta_tau_til = tau_til_vec[i] - Delta_tau_til
                if tau_til_vec[i] < 1e-6:
                    if tau_til_vec[i] > -1e-4:
                        tau_til_vec[i] = 1e-6
                        nu_til_vec[i] = 0.
                    else:
                        # raise ValueError("tau_til_vec[i] is negative!")
                        1

                logZ_til_vec[i] = np.log(Z) + 0.5 * log_2_pi + 0.5 * np.log(var_cav_i + 1. / tau_til_vec[i]) + 0.5 * (mu_cav_i - nu_til_vec[i])**2 / (var_cav_i + 1. / tau_til_vec[i])

                # update Sigma and mu
                si = Sigma[:, i]
                Sigma = Sigma - Delta_tau_til / (1 + Delta_tau_til * Sigma[i, i]) * np.outer(si, si)
                mu_vec = np.dot(Sigma, nu_til_vec)
            else:
                Delta_tau_til = tau_til_vec[i]
                tau_til_vec[i] = 1e6
                nu_til_vec[i] = y[i] * 1e6
                Delta_tau_til = tau_til_vec[i] - Delta_tau_til
                Z = 1.
                logZ_til_vec[i] = np.log(Z) + 0.5 * log_2_pi + 0.5 * np.log(var_cav_i + 1. / tau_til_vec[i]) + 0.5 * (
                            mu_cav_i - nu_til_vec[i]) ** 2 / (var_cav_i + 1. / tau_til_vec[i])
                # update Sigma and mu
                si = Sigma[:, i]
                Sigma = Sigma - Delta_tau_til / (1 + Delta_tau_til * Sigma[i, i]) * np.outer(si, si)
                mu_vec = np.dot(Sigma, nu_til_vec)


        # re-compute the approximate posterior parameters Sigma and mu
        iteration += 1
        S_til_sqrt = np.diag(np.sqrt(tau_til_vec))
        L = np.linalg.cholesky(np.eye(dim) + S_til_sqrt.dot(K).dot(S_til_sqrt))
        V = np.linalg.solve(L, S_til_sqrt.dot(K))
        Sigma = K - V.T.dot(V)
        mu_vec = Sigma.dot(nu_til_vec)

    # # standard version
    # term1 = -0.5 * np.log(np.linalg.det(K + Sigma)) - 0.5 * mu_vec.T.dot(np.linalg.inv(K + Sigma)).dot(mu_vec)
    # term2 = np.sum(logZ_hat_vec + 0.5 * np.log(var_cav_vec + 1. / tau_til_vec) + 0.5 * (mu_cav_vec - mu_vec)**2 / (var_cav_vec + 1. / tau_til_vec))
    # logZ = term1 + term2

    # stable version
    Sigma[:, :] = fix_singular_matrix(Sigma, what2fix="Fixing Sigma inside EP loop...")
    term1 = -0.5 * (mean.T.dot(np.linalg.solve(K, mean)) + np.log(np.linalg.det(K)))
    term2 = np.sum(logZ_til_vec + 0.5 * np.log(1 + tau_til_vec / tau_cav_vec) + 0.5 * (
            tau_til_vec * nu_cav_vec ** 2 / tau_cav_vec - 2 * nu_cav_vec * nu_til_vec - nu_til_vec ** 2) / (
                           tau_cav_vec + tau_til_vec))
    term3 = 0.5 * (mu_vec.T.dot(np.linalg.solve(Sigma, mu_vec)) + np.log(np.linalg.det(Sigma)))
    logZ = term1 + term2 + term3
    return mu_vec, Sigma, logZ_til_vec, logZ


def ep_probit(y, mean, K, iteration_num, scale=1.0, precision=1e-12):
    """
    This is an implementation of EP method follows the AAAI paper and GPML book, using probit function
    This function is not leveraged in CBOB code, please refer to ep_truncated for detailed configurations.
    y: observations
    mean: latent mean function
    K: covariance matrix
    iteration_num: iteration number of EP alg.
    scale: parameters of probit function for appriximation of step function
    precision: allowable computation errors
    """
    if y.ndim == 2:
        y = y[:, 0]
        mean = y * 0

    log_2_pi = np.log(2 * np.pi)
    # initialize the parameters
    dim = K.shape[0]
    nu_til_vec = np.zeros(dim)
    tau_til_vec = np.zeros(dim)
    mu_vec = mean * 0.
    Sigma = np.copy(K)

    mu_cav_vec = np.zeros(dim)
    var_cav_vec = np.zeros(dim)
    tau_cav_vec = np.zeros(dim)
    nu_cav_vec = np.zeros(dim)
    logZ_til_vec = np.zeros(dim)
    logZ_hat_vec = np.zeros(dim)
    nu_til_vec_old = np.zeros(dim)
    tau_til_vec_old = np.zeros(dim)

    norm_distribution = norm(loc=0., scale=1.)
    iteration = 0
    while iteration < iteration_num:
        for i in range(dim):
            # compute the approximate cavity parameters
            tau_cav_i = 1. / Sigma[i, i] - tau_til_vec[i]
            nu_cav_i = mu_vec[i] / Sigma[i, i] - nu_til_vec[i]
            if tau_cav_i < 1e-3:
                print("tau_cav_i is too small! (if optimizing, please omit it)")
                return None, None, None, 1e10
            tau_cav_vec[i] = tau_cav_i
            nu_cav_vec[i] = nu_cav_i

            # compute the marginal moments (only implement the probit function with scale)
            var_cav_i = 1. / tau_cav_i
            mu_cav_i = var_cav_i * nu_cav_i
            var_cav_vec[i] = var_cav_i
            mu_cav_vec[i] = mu_cav_i

            if y[i] > 0:
                z_i = y[i] * mu_cav_i / np.sqrt(1 * scale**2 + var_cav_i)
                if z_i < -35:
                    z_i = -35
                pdf = norm_distribution.pdf(z_i)
                cdf = norm_distribution.cdf(z_i)
                var_hat_i = var_cav_i - var_cav_i ** 2 * pdf / ((1 * scale**2 + var_cav_i) * cdf) * (z_i + pdf / cdf)
                mu_hat_i = mu_cav_i + y[i] * var_cav_i * pdf / (cdf * np.sqrt(1 * scale**2 + var_cav_i))
                Z_hat_i = cdf
            else:
                var_hat_i = 1. / (1e6 + tau_cav_i)
                mu_hat_i = var_hat_i * (y[i] * 1e6 + nu_cav_i)
                sum_var = 1e-6 + var_cav_i
                Z_hat_i = 1. / np.sqrt(2 * np.pi * sum_var) * np.exp(-0.5 * (y[i] - nu_cav_i / tau_cav_i)**2. / sum_var)

            if Z_hat_i < 1e-12:
                Z_hat_i = 1e-12

            logZ_hat_vec[i] = np.log(Z_hat_i)
            if var_hat_i < 1e-12:
                if var_hat_i > 0:
                    print("var_hat_i is too small!")
                    var_hat_i = 1e-12
                    Delta_tau_til = 0.
                else:
                    raise ValueError("var_hat_i is negative... Something goes wrong.")
            else:
                Delta_tau_til = 1 / var_hat_i - tau_cav_i - tau_til_vec[i]
            tau_til_vec[i] += Delta_tau_til
            nu_til_vec[i] = mu_hat_i / var_hat_i - nu_cav_i
            if tau_til_vec[i] < 1e-6:
                if tau_til_vec[i] > -1e-4:
                    tau_til_vec[i] = 1e-6
                    nu_til_vec[i] = 0.
                else:
                    raise ValueError("tau_til_vec[i] is negative!")
            if y[i] > 0:
                # logZ_til_vec[i] = np.log(Z_hat_i) + 0.5 * log_2_pi + 0.5 * np.log(var_cav_i + 1. / tau_til_vec[i]) + 0.5 * (mu_cav_i - nu_til_vec[i])**2 / (var_cav_i + 1. / tau_til_vec[i])
                logZ_til_vec[i] = np.log(
                    Z_hat_i * np.sqrt(2 * np.pi) * np.sqrt(var_cav_i + 1. / tau_til_vec[i]) * np.exp(
                        0.5 * (mu_cav_i - nu_til_vec[i] / tau_til_vec[i]) ** 2 / (var_cav_i + 1. / tau_til_vec[i])))
            else:
                logZ_til_vec[i] = 0.
            # update Sigma and mu
            si = Sigma[:, i]
            Sigma = Sigma - Delta_tau_til / (1 + Delta_tau_til * Sigma[i, i]) * np.outer(si, si)
            mu_vec = np.dot(Sigma, nu_til_vec)

        # re-compute the approximate posterior parameters Sigma and mu
        tau_diff = np.mean(np.square(tau_til_vec - tau_til_vec_old))
        v_diff = np.mean(np.square(nu_til_vec - nu_til_vec_old))

        tau_til_vec_old = tau_til_vec
        nu_til_vec_old = nu_til_vec
        iteration += 1
        S_til_sqrt = np.diag(np.sqrt(tau_til_vec))
        L = np.linalg.cholesky(np.eye(dim) + S_til_sqrt.dot(K).dot(S_til_sqrt))
        V = np.linalg.solve(L, S_til_sqrt.dot(K))
        Sigma = K - V.T.dot(V)
        mu_vec = Sigma.dot(nu_til_vec)

        # print("iteration num is: ", iteration - 1, "      the precious is, tau: ", tau_diff, " and nu: ", v_diff)


        if tau_diff < precision and v_diff < precision:
            break
    # Sigma[:, :] = fix_singular_matrix(Sigma, what2fix="Fixing Sigma inside EP loop...")

    print("############### EP computing finished. ##################")
    # GPy version 1
    mu_tilde = nu_til_vec / tau_til_vec
    mu_cav = nu_cav_vec / tau_cav_vec
    sigma2_sigma2tilde = 1. / tau_cav_vec + 1. / tau_til_vec
    logz_ep1 = np.sum((logZ_hat_vec + 0.5*np.log(2*np.pi) + 0.5*np.log(sigma2_sigma2tilde)
                         + 0.5*((mu_cav - mu_tilde)**2) / (sigma2_sigma2tilde)))

    # GPy version 2
    logz_ep2 = np.sum((
            logZ_hat_vec
            + 0.5 * np.log(2 * np.pi) + 0.5 * np.log(1 + tau_til_vec / tau_cav_vec)
            - 0.5 * ((nu_til_vec) ** 2 * 1. / (tau_cav_vec + tau_til_vec))
            + 0.5 * (nu_cav_vec * (((tau_til_vec / tau_cav_vec) * nu_cav_vec - 2.0 * nu_til_vec) * 1. / (
                tau_cav_vec + tau_til_vec)))
    ))

    # stable version
    # Sigma[:, :] = fix_singular_matrix(Sigma, what2fix="Fixing Sigma inside EP loop...")
    # term1 = -0.5 * (mean.T.dot(np.linalg.solve(K, mean)) + np.log(np.linalg.det(K)))
    # term2 = np.sum(logZ_til_vec + 0.5 * np.log(1 + tau_til_vec / tau_cav_vec) + 0.5 * (
    #         tau_til_vec * nu_cav_vec ** 2 / tau_cav_vec - 2 * nu_cav_vec * nu_til_vec - nu_til_vec ** 2) / (
    #                        tau_cav_vec + tau_til_vec))
    # term3 = 0.5 * (mu_vec.T.dot(np.linalg.solve(Sigma, mu_vec)) + np.log(np.linalg.det(Sigma)))
    # logZ = term1 + term2 + term3

    # # standard version
    mu_tilde = nu_til_vec / tau_til_vec
    mu_cav = nu_cav_vec / tau_cav_vec
    Sigma_tilde = np.diag(1. / (tau_til_vec + 1e-6))
    logz_ep3 = - 0.5 * np.log(np.linalg.det(Sigma_tilde + K)) - 0.5 * mu_tilde.T.dot(np.linalg.inv(K + Sigma_tilde)).dot(
        mu_tilde) + np.sum(
        logZ_hat_vec + 0.5 * np.log(1. / tau_cav_vec + 1. / tau_til_vec) + 0.5 * (mu_cav - mu_tilde) ** 2 / (
                    1. / tau_til_vec + 1. / tau_cav_vec)
    )

    return mu_vec, Sigma_tilde, np.sum(logZ_til_vec), logz_ep3


def fix_singular_matrix(singular_mat, verbosity=False, what2fix=None, val_min_deter=1e-200, val_max_cond=1e9):
    assert singular_mat.ndim == 2
    assert singular_mat.shape[0] == singular_mat.shape[1]

    # Corner case:
    cond_num = np.linalg.cond(singular_mat)
    deter = np.linalg.det(singular_mat)

    # Check positive definiteness:
    chol_ok = True
    try:
        np.linalg.cholesky(singular_mat)
    except Exception as inst:
        if verbosity == True:
            print(type(inst), inst.args)
        chol_ok = False

    # Check log(det)
    log_det_ok = True
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            np.log(deter)
    except Exception as inst:
        if verbosity == True:
            print(type(inst), inst.args)
        log_det_ok = False

    if cond_num <= val_max_cond and deter > val_min_deter and chol_ok == True and log_det_ok == True:
        return singular_mat
    else:
        pass
    print("@GaussianTools.fix_singular_matrix(): singular_mat needs to be fixed")
    if what2fix is not None: print("what2fix:",what2fix)

    # Get the order of magnitude of the largest eigenvalue in singular_mat, assuming all eigenvalues are positive:
    eigs_real = np.real(np.linalg.eigvals(singular_mat))
    largest_eig = np.amax(eigs_real)
    if largest_eig < 1e-310:
        max_ord = np.floor(np.log10(1e-310))
    else:
        max_ord = np.ceil(np.log10(largest_eig))

    # print("largest_eig: ",largest_eig)
    # print("max_ord: ",max_ord)

    # Get the order of magnitude of the smallest eigenvalue in singular_mat, assuming all eigenvalues are positive:
    smallest_eig = np.amin(eigs_real)
    if smallest_eig < 1e-310:
        min_ord = np.floor(np.log10(1e-310))
    else:
        min_ord = np.floor(np.log10(np.abs(smallest_eig)))

    # Initial factor:
    fac_init = min_ord * 2.

    if verbosity == True:
        print("\n[VERBOSITY]: @GaussianTools.fix_singular_matrix(): singular_mat needs to be fixed")
        print("cond_num:", cond_num)
        print("min_ord:", min_ord)
        print("max_ord:", max_ord)
        print("chol_ok:", chol_ok)
        print("log_det_ok:", log_det_ok)
        print("Before update:")
        print("==============")
        print("fac_init:", fac_init)
        print("order cond_num:", np.floor(np.log10(cond_num)))
        print("deter:", deter)
        print("eig:", np.linalg.eigvals(singular_mat))

    # Fix the matrix:
    Id = np.eye(singular_mat.shape[0])
    singular_mat_new = singular_mat
    c = 0
    singular = True
    fac = 10 ** (fac_init)
    while singular == True and fac_init + c < max_ord:

        # New factor:
        fac = 10 ** (fac_init + c)
        singular_mat_new[:, :] = singular_mat + fac * Id

        # Look for errors:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                np.linalg.cholesky(singular_mat_new)
                assert np.linalg.det(singular_mat_new) > val_min_deter
                np.log(np.linalg.det(singular_mat_new))
                assert np.linalg.cond(singular_mat_new) <= val_max_cond
        except Exception as inst:
            if verbosity == True:
                print(type(inst), inst.args)
            c += 1
        else:
            singular = False

    if verbosity == True:
        print("After update:")
        print("=============")
        print("fac:", fac)
        print("order cond_num:", np.floor(np.log10(np.linalg.cond(singular_mat_new))))
        print("deter:", np.linalg.det(singular_mat_new))
        print("eig:", np.linalg.eigvals(singular_mat_new))

    if singular == True:
        # pdb.set_trace()
        # raise ValueError("Matrix could not be fixed. Something is really wrong here...")
        # warnings.warn("Matrix could not be fixed. Something is really wrong here...")
        print("Matrix could not be fixed. Something is really wrong here...")  # Highest permission
    return singular_mat_new
