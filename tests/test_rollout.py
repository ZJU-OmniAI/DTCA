import pytest
import torch

from rollout.llada import LLaDAConfig, LLaDAModelLM
from rollout.llada_generate import generate_with_prefix_cache


@pytest.mark.parametrize("cache,horizon", [(False, None), (True, None), (True, 2)])
def test_padded_prompt_matches_single_prompt_and_cache_masks(cache, horizon):
    torch.manual_seed(17)
    model = LLaDAModelLM(LLaDAConfig(d_model=16, n_heads=2, n_layers=1, mlp_ratio=2,
                                    vocab_size=16, embedding_size=16, max_sequence_length=32,
                                    rope=True, block_type="llama", activation_type="silu"), init_params=True).eval()
    arguments = dict(steps=4, gen_length=4, block_length=2, temperature=0,
                     target="confidence", mask_id=15, further_horizon=horizon,
                     use_cache=cache, unmask_threshold=0.99)
    with torch.no_grad():
        single = generate_with_prefix_cache(model, torch.tensor([[1, 2]]), **arguments)
        batched = generate_with_prefix_cache(model, torch.tensor([[0, 1, 2], [3, 4, 5]]),
                                              attention_mask=torch.tensor([[0, 1, 1], [1, 1, 1]]),
                                              **arguments)
    assert torch.equal(single.sequences[0, -4:], batched.sequences[0, -4:])
    assert torch.allclose(single.token_probs[0, -4:], batched.token_probs[0, -4:], atol=1e-6)
    assert batched.sequences[:, -4:].ne(15).all()


def test_training_padding_preserves_bidirectional_context():
    from models.llada import LLaDAConfig as Config, LLaDAModelLM as Model

    torch.manual_seed(4)
    model = Model(Config(d_model=16, n_heads=2, n_layers=1, mlp_ratio=2,
                          vocab_size=16, embedding_size=16, max_sequence_length=32,
                          rope=True, block_type="llama", activation_type="silu"), init_params=True).eval()
    single = model(torch.tensor([[1, 2, 3]]), use_cache=False).logits
    padded = model(torch.tensor([[0, 1, 2, 3]]), attention_mask=torch.tensor([[0, 1, 1, 1]]),
                    use_cache=False).logits[:, 1:]
    changed = model(torch.tensor([[0, 1, 2, 4]]), attention_mask=torch.tensor([[0, 1, 1, 1]]),
                     use_cache=False).logits[:, 1:]
    assert torch.allclose(single, padded, atol=1e-6)
    assert not torch.allclose(padded[:, 0], changed[:, 0], atol=1e-8)
    padded.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_single_step_block_does_not_forward_after_completion():
    from types import SimpleNamespace

    class Model:
        def __init__(self):
            self.calls = 0

        def __call__(self, ids, **kwargs):
            self.calls += 1
            logits = torch.zeros(*ids.shape, 4)
            logits[..., 3] = 100  # Even a likely mask cannot be committed.
            return SimpleNamespace(logits=logits)

    model = Model()
    result = generate_with_prefix_cache(model, torch.tensor([[0]]), steps=1, gen_length=2,
                                         block_length=2, temperature=0, target="confidence",
                                         mask_id=3, further_horizon=None, use_cache=False,
                                         unmask_threshold=None)
    assert model.calls == result.nfe == 1
    assert result.sequences[:, 1:].ne(3).all()
