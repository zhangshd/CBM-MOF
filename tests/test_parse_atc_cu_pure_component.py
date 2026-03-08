import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.alignn.parse_atc_cu_pure_component import (  # noqa: E402
    BENCHMARK_MOF,
    STANDARD_COLUMNS,
    filter_atc_cu_pure_component,
)


class TestParseAtcCuPureComponent(unittest.TestCase):
    def test_filter_keeps_only_benchmark_pure_component_rows(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "MofName": BENCHMARK_MOF,
                    "GasName": "methane",
                    "Temperature[K]": 298.0,
                    "Pressure[bar]": 0.01,
                    "AllComponents": "methane",
                    "MoleculeFraction": 1.0,
                    "LoadingUnit": "mol/kg",
                    "AbsLoading": 0.12,
                    "ExcessLoading": 0.11,
                    "SimuDuration[h]": 0.5,
                    "FilePath": "/tmp/a",
                    "Notes": "",
                },
                {
                    "MofName": BENCHMARK_MOF,
                    "GasName": "N2",
                    "Temperature[K]": 298.0,
                    "Pressure[bar]": 0.01,
                    "AllComponents": "N2",
                    "MoleculeFraction": 1.0,
                    "LoadingUnit": "mol/kg",
                    "AbsLoading": 0.07,
                    "ExcessLoading": 0.06,
                    "SimuDuration[h]": 0.5,
                    "FilePath": "/tmp/b",
                    "Notes": "",
                },
                {
                    "MofName": BENCHMARK_MOF,
                    "GasName": "methane",
                    "Temperature[K]": 298.0,
                    "Pressure[bar]": 1.0,
                    "AllComponents": "N2_methane",
                    "MoleculeFraction": 0.2,
                    "LoadingUnit": "mol/kg",
                    "AbsLoading": 0.55,
                    "ExcessLoading": 0.50,
                    "SimuDuration[h]": 0.5,
                    "FilePath": "/tmp/c",
                    "Notes": "mixture row",
                },
                {
                    "MofName": BENCHMARK_MOF,
                    "GasName": "methane",
                    "Temperature[K]": 298.0,
                    "Pressure[bar]": 10.0,
                    "AllComponents": "methane",
                    "MoleculeFraction": 0.8,
                    "LoadingUnit": "mol/kg",
                    "AbsLoading": 1.55,
                    "ExcessLoading": 1.50,
                    "SimuDuration[h]": 0.5,
                    "FilePath": "/tmp/d",
                    "Notes": "bad mol fraction",
                },
                {
                    "MofName": "Other-MOF",
                    "GasName": "methane",
                    "Temperature[K]": 298.0,
                    "Pressure[bar]": 0.01,
                    "AllComponents": "methane",
                    "MoleculeFraction": 1.0,
                    "LoadingUnit": "mol/kg",
                    "AbsLoading": 0.22,
                    "ExcessLoading": 0.21,
                    "SimuDuration[h]": 0.5,
                    "FilePath": "/tmp/e",
                    "Notes": "",
                },
            ]
        )

        result = filter_atc_cu_pure_component(df)

        self.assertEqual(list(result.columns), STANDARD_COLUMNS)
        self.assertEqual(len(result), 2)
        self.assertEqual(set(result["MofName"]), {BENCHMARK_MOF})
        self.assertEqual(set(result["AllComponents"]), {"methane", "N2"})
        self.assertTrue((result["MoleculeFraction"] == 1.0).all())
        self.assertEqual(set(result["GasName"]), {"methane", "N2"})


if __name__ == "__main__":
    unittest.main()
