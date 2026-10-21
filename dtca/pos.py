"""spaCy POS projection onto completion token positions."""

from difflib import SequenceMatcher
from pathlib import Path

import torch

TAGS = ("ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SPACE", "SYM", "VERB", "X")
IDS = {tag: i for i, tag in enumerate(TAGS)}


class Tagger:
    def __init__(self, model="en_core_web_sm", nlp=None):
        if nlp is None:
            import spacy

            local = Path(__file__).resolve().parents[1] / "models" / model
            nlp = spacy.load(str(local) if local.exists() else model, disable=["ner"])
        self.nlp = nlp

    def project(self, tokenizer, tokens, mask):
        """Project exact retokenization matches; unaligned/special tokens remain X."""
        if tokens.shape != mask.shape or tokens.ndim != 2:
            raise ValueError("Token IDs and completion mask must be aligned matrices")
        result = torch.full_like(tokens, IDS["X"], dtype=torch.long)
        special = set(tokenizer.all_special_ids)
        for row in range(tokens.shape[0]):
            positions = [p for p in mask[row].bool().nonzero().flatten().tolist() if int(tokens[row, p]) not in special]
            ids = [int(tokens[row, p]) for p in positions]
            text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            spans = [(t.idx, t.idx + len(t.text), t.pos_) for t in self.nlp(text)]
            matcher = SequenceMatcher(a=ids, b=encoded["input_ids"], autojunk=False)
            for block in matcher.get_matching_blocks():
                for offset in range(block.size):
                    position = positions[block.a + offset]
                    start, end = encoded["offset_mapping"][block.b + offset]
                    if end <= start:
                        continue
                    # Whitespace is an explicit category in the 18-tag inventory.
                    if text[start:end].isspace():
                        result[row, position] = IDS["SPACE"]
                        continue
                    best, overlap = "X", 0
                    for left, right, tag in spans:
                        size = max(0, min(end, right) - max(start, left))
                        if size > overlap:
                            best, overlap = tag, size
                    result[row, position] = IDS.get(best, IDS["X"])
        return result


def aggregate(values, pos_ids, valid):
    """Accumulate token observations, including zero-valued contributions."""
    if not (values.shape == pos_ids.shape == valid.shape):
        raise ValueError("POS observations must align")
    ids, values = pos_ids[valid.bool()].long().cpu(), values[valid.bool()].double().cpu()
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("Contributions must be finite and nonnegative")
    sums = torch.zeros(len(TAGS), dtype=torch.float64).scatter_add_(0, ids, values)
    counts = torch.zeros(len(TAGS), dtype=torch.long).scatter_add_(0, ids, torch.ones_like(ids))
    return sums, counts
