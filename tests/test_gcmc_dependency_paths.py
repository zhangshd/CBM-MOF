import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.alignn import parse_atc_cu_pure_component as parse_atc  # noqa: E402
from src.alignn import parse_top20_pure_component as parse_top20  # noqa: E402


class TestGcmcDependencyPaths(unittest.TestCase):
    def test_pure_component_parsers_use_repo_local_gcmc_module(self) -> None:
        expected = REPO_ROOT / "src" / "gcmc"
        self.assertEqual(parse_top20.GCMC_SRC, expected)
        self.assertEqual(parse_atc.GCMC_SRC, expected)


if __name__ == "__main__":
    unittest.main()
