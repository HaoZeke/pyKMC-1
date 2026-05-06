import numpy as np 
import numpy.testing as npt
import pandas as pd

from pykmc.basins import FPTASelector, StatesConnectivity, solve_master_equation, BisectionSolver


class SequenceRng:
    def __init__(self, draws):
        self.draws = iter(draws)

    def random(self):
        return next(self.draws)


def _defective_reduced_ni_matrix():
    return np.array(
        [
            [0.01524846101964865, 0.0, 0.0, 0.0, 0.0],
            [-0.001906057627456081, 0.01524846101964865, 0.0, 0.0, 0.0],
            [-0.001906057627456081, 0.0, 0.01524846101964865, 0.0, 0.0],
            [0.0, -0.001906057627456081, 0.0, 0.01524846101964865, 0.0],
            [
                -0.011436345764736488,
                -0.01334240339219257,
                -0.015248461019648652,
                -0.01524846101964865,
                0.0,
            ],
        ]
    )


def _defective_ni_connectivity():
    table = StatesConnectivity()
    table.df = pd.DataFrame(
        [
            {"state": 0, "state_connexion": 1, "k_forward": 0.001906057627456081},
            {"state": 0, "state_connexion": 2, "k_forward": 0.001906057627456081},
            {"state": 0, "state_connexion": 4, "k_forward": 0.011436345764736488},
            {"state": 1, "state_connexion": 3, "k_forward": 0.001906057627456081},
            {"state": 1, "state_connexion": 5, "k_forward": 0.01334240339219257},
            {"state": 2, "state_connexion": 6, "k_forward": 0.015248461019648652},
            {"state": 3, "state_connexion": 7, "k_forward": 0.01524846101964865},
        ]
    )
    return table

class TestSolver : 

    def test_solve_master_equation(self, test_logger):

        #Mock reduced matric 
        M_abs_reduced = np.array([
            [+0.17 ,  -0.30 , - 0.20 ,  0.00],
            [- 0.05 , 0.48 ,  -0.25 ,  0.00],
            [- 0.10 ,  -0.10 , 0.45 ,  0.00],
            [- 0.02 ,  -0.08 ,  -0.00 ,  0.00]])
        
        p0 = np.array([1,0,0,0])
        t0 = 1.0/np.sum(np.diag(M_abs_reduced))

        test_logger.debug("Solve Master Equation for : ")
        test_logger.debug("M = \n {}".format(M_abs_reduced))
        test_logger.debug("p0 = \n {}".format(p0))
        test_logger.debug("t = {}".format(t0))
        p = solve_master_equation(M_abs_reduced, t0, p0, False)
        test_logger.debug("found p = \n {}".format(p))

        #computed with gnu octave
        res_expected = np.array([0.869014, 0.041671, 0.070828, 0.018488])
        test_logger.debug("Expected p = \n {}".format(res_expected))

        npt.assert_allclose(p, res_expected, rtol=1e-4)
        
        test_logger.debug("Solve Master Equation Using Sprectral Decomposition for : ")
        test_logger.debug("M = \n {}".format(M_abs_reduced))
        test_logger.debug("p0 = \n {}".format(p0))
        test_logger.debug("t = {}".format(t0))
        p = solve_master_equation(M_abs_reduced, t0, p0, True)
        test_logger.debug("found p = \n {}".format(p))

        test_logger.debug("Expected p = \n {}".format(res_expected))

        npt.assert_allclose(p, res_expected, rtol=1e-4)

    def test_find_texit(self, test_logger) : 
        
        M_abs_reduced = np.array([[ 1.89645002e-02,-9.48225009e-03,-9.48225009e-03, 0.00000000e+00],
 [-9.48225009e-03, 1.89645002e-02,-9.48225009e-03, 0.00000000e+00],
 [-9.48225009e-03,-9.48225009e-03, 1.89645002e-02, 0.00000000e+00],
 [-2.83934789e-10,-2.83934789e-10,-2.83934789e-10, 0.00000000e+00]])
        
        p0 = np.array([1,0,0,0])
        r = 0.9

        solver = BisectionSolver(M=M_abs_reduced, p0=p0, r=r, spectral_decomposition=True) 

        test_logger.debug("Find t_exit for r = {}".format(r))
        test_logger.debug("With M = \n {}".format(M_abs_reduced))
        test_logger.debug("And p0 = \n {}".format(p0))

        res = solver.solve()

        if res.is_ok() : 
            t_exit = res.ok_value().t_exit
            test_logger.debug("Find t_exit = {}ps".format(t_exit))
        else : 
            err = res.err_value()
            test_logger.debug("Err while searching t_exit : {}".format(err))

    def test_find_texit_without_spectral_decomposition(self):
        M_abs_reduced = np.array(
            [
                [2.0, 0.0],
                [-2.0, 0.0],
            ]
        )
        p0 = np.array([1.0, 0.0])

        solver = BisectionSolver(
            M=M_abs_reduced,
            p0=p0,
            r=0.5,
            spectral_decomposition=False,
        )

        result = solver.solve()

        assert result.is_ok()
        assert result.ok_value().t_exit > 0.0

    def test_spectral_exit_solver_falls_back_on_defective_generator(self):
        M_abs_reduced = _defective_reduced_ni_matrix()
        p0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

        solver = BisectionSolver(
            M=M_abs_reduced,
            p0=p0,
            r=0.5,
            spectral_decomposition=True,
        )
        result = solver.solve()

        assert result.is_ok()
        t_exit = result.ok_value().t_exit
        p_abs = solve_master_equation(
            M_abs_reduced,
            t_exit,
            p0,
            spectral_decomposition=False,
        )[-1]
        npt.assert_allclose(p_abs, 0.5, rtol=1.0e-3, atol=1.0e-6)

    def test_fpta_selector_matches_master_equation_on_defective_generator(self):
        table = _defective_ni_connectivity()
        selector = FPTASelector(rng=SequenceRng([0.5, 0.5]))

        result = selector.select_from_connectivity(table)

        assert result.is_ok()
        value = result.ok_value()
        n_transient = len(set(table.df["state"]))
        p0 = np.zeros(len(selector.M_abs))
        p0[0] = 1.0
        p = solve_master_equation(
            selector.M_abs,
            value.t_exit,
            p0,
            spectral_decomposition=False,
        )
        p_absorbing = p[n_transient:]
        total_absorbed = float(np.sum(p_absorbing))
        conditional = p_absorbing / total_absorbed
        expected_exit = n_transient + int(np.searchsorted(np.cumsum(conditional), 0.5))

        npt.assert_allclose(total_absorbed, 0.5, rtol=1.0e-3, atol=1.0e-6)
        assert value.exit_state == expected_exit
