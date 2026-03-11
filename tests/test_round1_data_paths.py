import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.alignn import plot_bkt_paper_figures as plot_bkt  # noqa: E402

if "seaborn" not in sys.modules:
    sys.modules["seaborn"] = types.ModuleType("seaborn")
if "sklearn" not in sys.modules:
    sklearn_mod = types.ModuleType("sklearn")
    metrics_mod = types.ModuleType("metrics")
    sklearn_mod.metrics = metrics_mod
    sys.modules["sklearn"] = sklearn_mod
    sys.modules["sklearn.metrics"] = metrics_mod

from src.alignn import visualize_gcmc_validation as vis_gcmc  # noqa: E402


class TestRound1DataPaths(unittest.TestCase):
    def test_round1_benchmark_csvs_use_repo_local_results_tree(self) -> None:
        ads_expected = (
            REPO_ROOT
            / "results"
            / "cbm_screening"
            / "gcmc_round1_DreidingTraPPEJson"
            / "raspa3_parsed_results_0911.csv"
        )
        widom_expected = (
            REPO_ROOT
            / "results"
            / "cbm_screening"
            / "widom_round1_DREIDING"
            / "widom_results_0911.csv"
        )
        self.assertEqual(plot_bkt.TRAINING_ADS_R1_CSV, ads_expected)
        self.assertEqual(plot_bkt.TRAINING_WIDOM_R1_CSV, widom_expected)
        self.assertEqual(vis_gcmc.TRAINING_ADS_R1_CSV, ads_expected)
        self.assertEqual(vis_gcmc.TRAINING_WIDOM_R1_CSV, widom_expected)


if __name__ == "__main__":
    unittest.main()
