"""Checkpoint weights/tokenizer must reload through the evaluation model path."""

from pathlib import Path

from accelerate import Accelerator
from omegaconf import OmegaConf
from tokenizers import Tokenizer, models
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import torch

from models.llada import LLaDAConfig, LLaDAModelLM
from rollout.llada import LLaDAModelLM as Rollout
from trainer.utils import save_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def test_final_checkpoint_reloads_identical_train_and_rollout_logits(tmp_path):
    torch.manual_seed(31)
    settings = OmegaConf.load(ROOT / "configs/dtca.yaml")
    settings.experiment.project = str(tmp_path)
    settings.experiment.checkpoint_dir = str(tmp_path / "model")
    settings.model.pretrained_model = str(ROOT / "models/llada")
    config = LLaDAConfig(d_model=16, n_heads=2, n_layers=1, mlp_ratio=2,
                          vocab_size=16, embedding_size=16, max_sequence_length=32,
                          rope=True, block_type="llama", activation_type="silu")
    model = LLaDAModelLM(config, init_params=True).eval()
    backend = Tokenizer(models.WordLevel({"<unk>": 0, "<pad>": 1, **{str(i): i for i in range(2, 16)}},
                                         unk_token="<unk>"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>")
    accelerator = Accelerator(cpu=True)
    save_checkpoint(model, tokenizer, settings, accelerator, "checkpoint-final")
    checkpoint = tmp_path / "model/checkpoint-final"
    restored = LLaDAModelLM.from_pretrained(checkpoint).eval()
    rollout = Rollout.from_pretrained(checkpoint).eval()
    ids = torch.tensor([[1, 2, 3, 15]])
    attention = torch.tensor([[0, 1, 1, 1]])
    with torch.no_grad():
        expected = model(ids, attention_mask=attention, use_cache=False).logits
        assert torch.allclose(restored(ids, attention_mask=attention, use_cache=False).logits, expected, atol=1e-7)
        assert torch.allclose(rollout(ids, attention_mask=attention, use_cache=True).logits, expected, atol=1e-7)
    loaded = AutoTokenizer.from_pretrained(checkpoint)
    assert loaded.get_vocab() == tokenizer.get_vocab()
    assert loaded.pad_token_id == tokenizer.pad_token_id
    assert (checkpoint / "modeling_llada.py").is_file()
    assert (checkpoint / "configuration_llada.py").is_file()
