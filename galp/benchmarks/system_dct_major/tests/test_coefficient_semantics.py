from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

from galp.benchmarks.system_dct_major import verify_coefficient_semantics as oracle


class CoefficientSemanticsTest(unittest.TestCase):
    def verify(self, error=0.0):
        contract = {
            "dataset": {"sample_manifest": "samples.json"},
            "execution": {"device": "cpu"},
            "models": {"rgbnomore_root": "upstream", "dct": {"checkpoint": "weights.pth"}},
            "pipelines": {"dct_major_pushdown": {
                "manifest": "manifest.bin", "torch_binding_dir": "binding",
                "block_major_access_dir": "access",
            }},
            "semantic_validation": {
                "input_max_abs": 1e-3, "input_mean_abs": 1e-5,
                "logit_cosine_min": 0.999, "semantic_top1_agreement_min": 1.0,
            },
        }
        reader = MagicMock()
        observed = []
        scopes = []

        def pipeline(_profile, *, dct_coeffs):
            selection = oracle.resolve_coefficient_selection(dct_coeffs)
            observed.append(dct_coeffs)
            values = torch.zeros(2, 64)
            values[:, selection["resolved_natural_indices"]] = 1
            batch = SimpleNamespace(global_image_ids=[3, 7], y=values + error, cbcr=values)
            scope = MagicMock()
            scopes.append(scope)
            scope.start.return_value.__enter__.return_value = iter([batch])
            return scope

        def preprocess(y, cbcr, quant, mask):
            values = mask.reshape(1, 1, 64).float().expand(1, 2, 64)
            return values, values

        reader.pipeline.side_effect = pipeline
        with ExitStack() as stack:
            stack.enter_context(patch.dict(oracle.os.environ, {}, clear=False))
            stack.enter_context(patch.object(oracle, "load_contract", return_value=contract))
            stack.enter_context(patch.object(oracle, "load_sample_manifest", return_value=[
                {"galp_image_id": 3}, {"galp_image_id": 7}, {"galp_image_id": 8}, {"galp_image_id": 9},
            ]))
            stack.enter_context(patch.object(oracle, "DirectDctReader", return_value=reader))
            stack.enter_context(patch.object(oracle, "RawDctDataset", return_value=[
                (torch.ones(64), torch.ones(64), torch.ones(64)),
                (torch.ones(64), torch.ones(64), torch.ones(64)),
            ]))
            stack.enter_context(patch.object(oracle, "BatchedDctPreprocessor", return_value=preprocess))
            stack.enter_context(patch.object(oracle, "build_workload_model", return_value=lambda y, cbcr: y))
            write = stack.enter_context(patch.object(oracle, "write_json"))
            oracle.verify(Path("contract.json"), Path("output.json"), 2)
        for scope in scopes:
            scope.start.assert_called_once_with([[3, 7], [8, 9]])
        return observed, write.call_args.args[1]

    def test_checks_all_four_selections_against_raw_mask_reference(self):
        observed, result = self.verify()
        self.assertEqual(observed, ["all", "first:32", "first:16", "list:0,2,5,9"])
        self.assertTrue(result["ok"])
        self.assertEqual(result["samples"], 2)
        self.assertTrue(all(case["top1_agreement"] == 1.0 for case in result["cases"]))

    def test_rejects_native_input_different_from_raw_mask_reference(self):
        with self.assertRaises(AssertionError):
            self.verify(error=0.1)


if __name__ == "__main__":
    unittest.main()
