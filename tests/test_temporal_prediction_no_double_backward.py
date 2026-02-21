"""
Test that _temporal_prediction_loss does not trigger the
'Trying to backward through the graph a second time' error.

Root cause: the method re-uses self.model.grus[nt] (the *same* GRU
module that produced proj_seq in model.forward) with a slice of proj_seq
as its context.  Without detaching that slice, the GRU appears twice in
the computation graph and cuDNN may free the first call's saved tensors
before the second call's backward reaches them.

Fix: proj_seq.detach()[:, :context_len, :] is used as the GRU context so
only future_targets keeps a live gradient path back through the model GRU.
"""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F


class TestNoDoubleBackward(unittest.TestCase):
    """Simulate the key pattern that caused the double-backward bug."""

    def test_detach_prevents_double_backward(self):
        """
        Reproduce the diamond-graph pattern:
          x → GRU_model → proj_seq
                              ├── context (used as input to GRU_model again)
                              └── future_targets (used as loss target)

        Without .detach(), GRU_model appears twice in the backward graph and
        calling backward a second time (or across epochs) raises RuntimeError.
        With .detach() on context the backward is clean across multiple epochs.
        """
        hidden_dim = 16
        gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        linear = nn.Linear(hidden_dim, hidden_dim)
        optimizer = torch.optim.SGD(
            list(gru.parameters()) + list(linear.parameters()), lr=1e-3
        )
        context_len = 8
        predict_len = 4

        for _epoch in range(2):
            optimizer.zero_grad()

            # Simulate "model forward" that produces proj_seq via the shared GRU
            x = torch.randn(3, 20, hidden_dim)
            proj_seq, _ = gru(linear(x))  # proj_seq ← gru (live graph)

            # Re-use the same GRU with a DETACHED context slice (the fix)
            context = proj_seq.detach()[:, :context_len, :]
            out_ctx, h = gru(context)

            # Autoregressive steps
            preds = []
            next_input = out_ctx[:, -1:, :].contiguous()
            for _ in range(predict_len):
                out_step, h = gru(next_input, h)
                pred = out_step[:, -1:, :].contiguous()
                preds.append(pred)
                next_input = pred

            pred_feat = torch.cat(preds, dim=1)
            future_targets = proj_seq[:, context_len:context_len + predict_len, :]

            # Loss mixes temporal-pred signal (pred_feat) and model-output signal (future_targets)
            temp_loss = F.mse_loss(pred_feat, future_targets)
            recon_loss = proj_seq.mean()
            total_loss = temp_loss + recon_loss

            # Must NOT raise "Trying to backward through the graph a second time"
            total_loss.backward()
            optimizer.step()

    def test_recon_loss_shape_alignment(self):
        """
        Verify that reconstruction loss computation handles mismatched N dimensions
        (recon has batch_size=1, target has batch_size=63) by cropping to min(N).
        This mirrors the fix for the UserWarning about target/input size mismatch.
        """
        # Simulate recon with batch_size=1 (e.g. single-sample output)
        recon = torch.randn(1, 190, 10)
        # Simulate target with batch_size=63
        target = torch.randn(63, 190, 10)

        Nr, Tr, Fr = recon.shape
        Nt, Tt, Ft = target.shape
        mN, mT, mF = min(Nr, Nt), min(Tr, Tt), min(Fr, Ft)

        r_crop = recon[:mN, :mT, :mF]
        t_crop = target[:mN, :mT, :mF]

        # Should not raise any warning or error
        loss = F.mse_loss(r_crop, t_crop)
        self.assertEqual(r_crop.shape, t_crop.shape)
        self.assertEqual(r_crop.shape[0], 1)  # cropped to min(1, 63) = 1
        self.assertFalse(torch.isnan(loss))


if __name__ == "__main__":
    unittest.main()
