from types import SimpleNamespace

import torch

from dtca.pos import IDS, Tagger


class Tokens:
    all_special_ids = [99]

    def decode(self, ids, **kwargs):
        return "go  now"

    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 2, 3], "offset_mapping": [(0, 2), (2, 4), (4, 7)]}


def test_exact_token_spans_space_and_specials():
    nlp = lambda text: [SimpleNamespace(idx=0, text="go", pos_="VERB"), SimpleNamespace(idx=4, text="now", pos_="ADV")]
    tagger = Tagger(nlp=nlp)
    result = tagger.project(Tokens(), torch.tensor([[1, 2, 3, 99, 88]]), torch.ones(1, 5))
    assert result[0].tolist() == [IDS["VERB"], IDS["SPACE"], IDS["ADV"], IDS["X"], IDS["X"]]


def test_byte_level_tokenizer_offsets_with_unicode_numbers_and_whitespace():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    text = "Take 2.5 café 😊\n?"
    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator([text], trainers.BpeTrainer(vocab_size=258, special_tokens=["<unk>", "<pad>"],
                                                          initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    tokens = torch.tensor([[tokenizer.pad_token_id] + encoded["input_ids"] + [tokenizer.pad_token_id]])
    spans = [(0, "Take", "VERB"), (5, "2.5", "NUM"), (9, "café", "NOUN"),
             (14, "😊", "SYM"), (16, "?", "PUNCT")]
    nlp = lambda _: [SimpleNamespace(idx=start, text=word, pos_=tag) for start, word, tag in spans]
    result = Tagger(nlp=nlp).project(tokenizer, tokens, torch.ones_like(tokens))
    assert result[0, 0] == result[0, -1] == IDS["X"]
    for actual, (start, end) in zip(result[0, 1:-1].tolist(), encoded["offset_mapping"]):
        expected = "SPACE" if text[start:end].isspace() else next(tag for left, word, tag in spans if left <= start < left+len(word))
        assert actual == IDS[expected]
