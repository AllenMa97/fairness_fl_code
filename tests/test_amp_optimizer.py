import unittest

import torch

from algorithm.Optimizers import BERTCLF_Optimizer


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for AMP regression")
class BERTCLFOptimizerAMPTest(unittest.TestCase):
    def test_grad_scaler_steps_wrapper_and_updates_parameters(self):
        device = torch.device("cuda:0")
        model = torch.nn.Linear(4, 2).to(device)
        optimizer = BERTCLF_Optimizer(
            method="sgd",
            learning_rate=0.1,
            max_grad_norm=0,
        )
        optimizer.set_parameters(model.named_parameters())
        scaler = torch.amp.GradScaler("cuda")
        inputs = torch.randn(8, 4, device=device)
        targets = torch.randn(8, 2, device=device)
        before = [parameter.detach().clone() for parameter in model.parameters()]

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            loss = torch.nn.functional.mse_loss(model(inputs), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        self.assertIs(optimizer.param_groups, optimizer.optimizer.param_groups)
        self.assertIs(optimizer.state, optimizer.optimizer.state)
        self.assertTrue(
            any(
                not torch.equal(old, new)
                for old, new in zip(before, model.parameters())
            )
        )


if __name__ == "__main__":
    unittest.main()
