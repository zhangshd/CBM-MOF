import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.alignn.fit_pure_component_isotherms import (  # noqa: E402
    DEFAULT_INPUT_COLUMNS,
    flatten_selected_fit,
    load_input_csvs,
    select_unified_model_for_mof,
)


class TestFitPureComponentIsotherms(unittest.TestCase):
    def test_load_input_csvs_concatenates_multiple_standard_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            csv_a = tmpdir / "a.csv"
            csv_b = tmpdir / "b.csv"

            row_template = {
                "MofName": "MOF-A",
                "GasName": "methane",
                "Temperature[K]": 298.0,
                "Pressure[bar]": 0.1,
                "AllComponents": "methane",
                "MoleculeFraction": 1.0,
                "LoadingUnit": "mol/kg",
                "AbsLoading": 1.23,
                "ExcessLoading": 1.10,
                "SimuDuration[h]": 0.5,
                "FilePath": "/tmp/path",
                "Notes": "",
            }
            pd.DataFrame([row_template]).to_csv(csv_a, index=False)
            row_b = dict(row_template)
            row_b["MofName"] = "MOF-B"
            row_b["GasName"] = "N2"
            row_b["AllComponents"] = "N2"
            pd.DataFrame([row_b]).to_csv(csv_b, index=False)

            result = load_input_csvs([csv_a, csv_b])

            self.assertEqual(list(result.columns), DEFAULT_INPUT_COLUMNS)
            self.assertEqual(len(result), 2)
            self.assertEqual(set(result["MofName"]), {"MOF-A", "MOF-B"})

    def test_select_unified_model_prefers_higher_mean_ranking_score(self) -> None:
        methane_langmuir = {
            "fit_quality": {"R2": 0.98, "ranking_score": 0.80},
            "fitted_isotherm": {"isotherm_model": {"name": "Langmuir"}},
        }
        n2_langmuir = {
            "fit_quality": {"R2": 0.97, "ranking_score": 0.79},
            "fitted_isotherm": {"isotherm_model": {"name": "Langmuir"}},
        }
        methane_dsl = {
            "fit_quality": {"R2": 0.981, "ranking_score": 0.60},
            "fitted_isotherm": {"isotherm_model": {"name": "DSLangmuir"}},
        }
        n2_dsl = {
            "fit_quality": {"R2": 0.971, "ranking_score": 0.59},
            "fitted_isotherm": {"isotherm_model": {"name": "DSLangmuir"}},
        }

        model_results = {
            "Langmuir": {
                "methane_298.0": methane_langmuir,
                "N2_298.0": n2_langmuir,
            },
            "DSLangmuir": {
                "methane_298.0": methane_dsl,
                "N2_298.0": n2_dsl,
            },
        }

        selected_model, summary = select_unified_model_for_mof(model_results)

        self.assertEqual(selected_model, "Langmuir")
        self.assertEqual(summary["n_gases_fit"], 2)
        self.assertAlmostEqual(summary["mean_ranking_score"], 0.795)

    def test_flatten_selected_fit_maps_model_specific_parameters(self) -> None:
        result = {
            "fitted_isotherm": {
                "adsorbate": "methane",
                "material": "MOF-A",
                "temperature": 298.0,
                "pressure_unit": "bar",
                "loading_unit": "mmol",
                "isotherm_model": {
                    "name": "Langmuir",
                    "parameters": {"K": 1.2, "n_m": 5.3},
                },
            },
            "experimental_data": {
                "pressures": [0.1, 1.0, 10.0],
                "loadings": [0.5, 2.0, 4.8],
            },
            "fit_quality": {
                "R2": 0.99,
                "MAE": 0.02,
                "RMSE": 0.03,
                "ranking_score": 0.8,
            },
        }
        selection_summary = {
            "selected_model": "Langmuir",
            "mean_ranking_score": 0.79,
            "mean_r2": 0.98,
            "n_gases_fit": 2,
        }

        row = flatten_selected_fit("MOF-A", "methane_298.0", result, selection_summary)

        self.assertEqual(row["selected_model"], "Langmuir")
        self.assertEqual(row["bkt_isomodel"], "Langmuir-Freundlich")
        self.assertEqual(row["K"], 1.2)
        self.assertEqual(row["n_m"], 5.3)
        self.assertIsNone(row["K1"])
        self.assertEqual(row["n_points"], 3)


if __name__ == "__main__":
    unittest.main()
