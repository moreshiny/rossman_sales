import unittest
import numpy as np
from sklearn.pipeline import Pipeline
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

from models import rmspe
from models import define_pipelines


class TestRMSPE(unittest.TestCase):

    def test_rmspe_returns_0_for_identical_arrays(self):
        prediction = np.ones((5))
        target = np.ones((5))
        self.assertAlmostEqual(rmspe(prediction, target), 0.0)

    def test_rmspe_returns_100_for_different_by_1_array(self):
        prediction = np.ones((5))
        prediction *= 2
        target = np.ones((5))
        self.assertAlmostEqual(rmspe(prediction, target), 100.0)

    def test_rmspe_returns_44_for_1_in_5_different_by_1_array(self):
        prediction = np.ones((5))
        prediction[0] = 2
        target = np.ones((5))
        self.assertAlmostEqual(rmspe(prediction, target), 44.72135954999579)

    def test_rmspe_returns_numpy_float64(self):
        prediction = np.ones((5))
        target = np.ones((5))
        self.assertAlmostEqual(type(rmspe(prediction, target)), np.float64)


class TestDefinePipelines(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

        self.xg_settings = dict(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.2,
            random_state=42,
            n_jobs=-1,
        )

        self.pipeline_input_1 = [
            (XGBRegressor, self.xg_settings),
        ]

        self.rf_settings = dict(
            n_estimators=50,
            max_depth=50,
            random_state=42,
            n_jobs=-1,
        )

        self.pipeline_input_2 = [
            (XGBRegressor, self.xg_settings),
            (RandomForestRegressor, self.rf_settings)
        ]

    def test_old_define_pipelines_sets_values_provided(self):
        # TODO remove this once there are no more code dependencies on legacy syntax
        pipes = define_pipelines(self.xg_settings)
        for pipe in pipes:
            for key in self.xg_settings.keys():
                self.assertEqual(
                    (key, pipe['model'].get_params()[key]),
                    (key, self.xg_settings[key])
                )

    def test_define_pipelines_returns_tuple(self):
        pipes = define_pipelines(self.pipeline_input_1)
        self.assertEquals(type(pipes), tuple)

    def test_define_pipelines_returns_one_pipe(self):
        pipes = define_pipelines(self.pipeline_input_1)
        self.assertEqual(len(pipes), 1)

    def test_define_pipelines_returns_tuple_of_dicts(self):
        pipes = define_pipelines(self.pipeline_input_1)
        for pipe in pipes:
            self.assertEqual(type(pipe), Pipeline)

    def test_define_pipelines_sets_values_provided(self):
        pipes = define_pipelines(self.pipeline_input_1)
        for pipe in pipes:
            for key in self.xg_settings.keys():
                self.assertEqual(
                    (key, pipe['model'].get_params()[key]),
                    (key, self.xg_settings[key])
                )

    def test_define_pipelines_sets_values_provided_2_models(self):
        pipes = define_pipelines(self.pipeline_input_2)
        for key in self.xg_settings.keys():
            self.assertEqual(
                (key, pipes[0]['model'].get_params()[key]),
                (key, self.xg_settings[key])
            )
        for key in self.rf_settings.keys():
            self.assertEqual(
                (key, pipes[1]['model'].get_params()[key]),
                (key, self.rf_settings[key])
            )

if __name__ == "__main__":
    unittest.main()
