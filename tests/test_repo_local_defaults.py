import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.alignn import filter_stable_candidates as stable_filter  # noqa: E402
from src.alignn import parse_pure_component_results as parse_pure  # noqa: E402
from src.figures import fig_process_validation as plot_bkt  # noqa: E402


class TestRepoLocalDefaults(unittest.TestCase):
    def test_defaults_use_repo_local_paths(self) -> None:
        self.assertEqual(
            stable_filter.STABILITY_CSV,
            REPO_ROOT / "data" / "processed" / "stabilities" / "infer_results_mofsnn.csv",
        )
        self.assertEqual(
            parse_pure.REPO_ROOT / "results" / "cbm_screening" / "gcmc_ATC-Cu_DreidingTraPPEJson",
            REPO_ROOT / "results" / "cbm_screening" / "gcmc_ATC-Cu_DreidingTraPPEJson",
        )
        self.assertEqual(
            plot_bkt.get_default_fig_dir(REPO_ROOT / "results" / "alignn" / "model_ep150"),
            REPO_ROOT / "results" / "alignn" / "model_ep150" / "figures",
        )


if __name__ == "__main__":
    unittest.main()
