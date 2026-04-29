import unittest
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

    register_cuda_ci(est_time=2, suite="stage-b-test-1-gpu-large")
    register_amd_ci(est_time=2, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipIf(torch is None, "torch is required for logprob normalization tests")
class TestLogprobNormalization(unittest.TestCase):
    def test_detokenize_top_logprobs_accepts_tensor_rows(self):
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        tokenizer_manager = TokenizerManager.__new__(TokenizerManager)

        ret = tokenizer_manager.detokenize_top_logprobs_tokens(
            [
                torch.tensor([-0.1, -0.2], dtype=torch.float64),
                torch.empty(0, dtype=torch.float64),
                None,
            ],
            [torch.tensor([10, 11]), torch.empty(0, dtype=torch.int64), None],
            decode_to_text=False,
        )

        self.assertEqual(len(ret), 3)
        self.assertEqual(len(ret[0]), 2)
        self.assertAlmostEqual(ret[0][0][0], -0.1)
        self.assertEqual(ret[0][0][1:], (10, None))
        self.assertAlmostEqual(ret[0][1][0], -0.2)
        self.assertEqual(ret[0][1][1:], (11, None))
        self.assertIsNone(ret[1])
        self.assertIsNone(ret[2])

    def test_add_logprob_return_values_materializes_top_logprobs_tensors(self):
        from sglang.srt.managers.scheduler_output_processor_mixin import (
            SchedulerOutputProcessorMixin,
        )

        processor = SchedulerOutputProcessorMixin()
        req = SimpleNamespace(
            top_logprobs_num=2,
            token_ids_logprob=None,
            output_token_logprobs_val=[],
            output_token_logprobs_idx=[],
            output_top_logprobs_val=[],
            output_top_logprobs_idx=[],
            output_token_ids_logprobs_val=[],
            output_token_ids_logprobs_idx=[],
            input_token_logprobs_val=None,
            input_token_logprobs_idx=None,
            input_top_logprobs_val=None,
            input_top_logprobs_idx=None,
            input_token_ids_logprobs_val=None,
            input_token_ids_logprobs_idx=None,
        )
        output = SimpleNamespace(
            next_token_logprobs=None,
            next_token_top_logprobs_val=[
                torch.tensor([-0.1, -0.2], dtype=torch.float64)
            ],
            next_token_top_logprobs_idx=[torch.tensor([10, 11])],
            next_token_token_ids_logprobs_val=None,
        )

        processor.add_logprob_return_values(
            i=0,
            req=req,
            pt=0,
            next_token_ids=[42],
            num_input_logprobs=0,
            output=output,
        )

        self.assertEqual(req.output_top_logprobs_val, [[-0.1, -0.2]])
        self.assertEqual(req.output_top_logprobs_idx, [[10, 11]])


if __name__ == "__main__":
    unittest.main()
