import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.figures import fig_process_validation as plot_bkt  # noqa: E402


class TestPlotBktDefaultFigDir(unittest.TestCase):
    def test_default_fig_dir_follows_model_dir(self) -> None:
        model_dir = REPO_ROOT / "results" / "alignn" / "model_ep220"
        self.assertEqual(plot_bkt.get_default_fig_dir(model_dir), model_dir / "figures")


if __name__ == "__main__":
    unittest.main()
