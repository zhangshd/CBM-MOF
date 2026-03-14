import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.alignn.run_breakthrough import ATC_CU_CIF  # noqa: E402


class TestBktDependencyPaths(unittest.TestCase):
    def test_atc_cu_cif_uses_repo_local_gcmc_example(self) -> None:
        expected = (
            REPO_ROOT
            / "src"
            / "gcmc"
            / "examples"
            / "dup_demo_ATC-Cu"
            / "CoRE-2020[Cu][pts]3[ASR]1.cif"
        )
        self.assertEqual(ATC_CU_CIF, expected)


if __name__ == "__main__":
    unittest.main()
