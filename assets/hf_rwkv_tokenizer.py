# coding=utf-8
"""Minimal Hugging Face tokenizer shim for RWKV vocab files."""

from __future__ import annotations

from ast import literal_eval
import os
from typing import TYPE_CHECKING, List, Optional, Tuple

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import logging


if TYPE_CHECKING:
    pass

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "rwkv_vocab_v20260603.txt",
}


class TRIE:
    __slots__ = tuple("ch,to,values,front".split(","))
    to: list
    values: set

    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for _ in range(256)]
        self.values = set()
        self.front = front

    def add(self, key: bytes, idx: int = 0, val=None):
        if idx == len(key):
            if val is None:
                val = key
            self.values.add(val)
            return self
        ch = key[idx]
        if self.to[ch] is None:
            self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx + 1, val=val)

    def find_longest(self, key: bytes, idx: int = 0):
        node = self
        ch = key[idx]
        best = None
        while node.to[ch] is not None:
            node = node.to[ch]
            idx += 1
            if node.values:
                best = idx, node.values
            if idx == len(key):
                break
            ch = key[idx]
        if best is None:
            raise ValueError("Failed to match an RWKV token.")
        return best


class RWKV_TOKENIZER:
    def __init__(self, file_name):
        self.idx2token = {}
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            idx = int(line[: line.index(" ")])
            token = literal_eval(line[line.index(" ") : line.rindex(" ")])
            token = token.encode("utf-8") if isinstance(token, str) else token
            assert isinstance(token, bytes)
            self.idx2token[idx] = token

        self.token2idx = {token: int(idx) for idx, token in self.idx2token.items()}
        self.root = TRIE()
        for token, token_id in self.token2idx.items():
            self.root.add(token, val=(token, token_id))

    def encode_bytes(self, src: bytes):
        idx = 0
        tokens = []
        while idx < len(src):
            next_idx, values = self.root.find_longest(src, idx)
            _, token_id = next(iter(values))
            tokens.append(token_id)
            idx = next_idx
        return tokens

    def decode_bytes(self, tokens):
        return b"".join(self.idx2token[token] for token in tokens)

    def encode(self, src):
        if isinstance(src, str):
            return [self.encode_bytes(src.encode("utf-8"))]
        if isinstance(src, list):
            return [self.encode_bytes(item.encode("utf-8")) for item in src]
        raise TypeError(f"Unsupported input type: {type(src)!r}")

    def decode(self, tokens):
        return [self.decode_bytes(batch).decode("utf-8") for batch in tokens]


class RwkvTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        **kwargs,
    ):
        if not os.path.isfile(vocab_file):
            raise ValueError(f"Can't find RWKV vocab file at path '{vocab_file}'.")

        self.add_bos_token = bool(kwargs.pop("add_bos_token", False))
        self.trie_tokenizer = RWKV_TOKENIZER(vocab_file)
        self.encoder = self.trie_tokenizer.token2idx
        self.decoder = {v: k for k, v in self.encoder.items()}
        self._added_tokens_decoder = {0: AddedToken(str(bos_token))}
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        vocab = dict(sorted(self.encoder.items(), key=lambda item: item[1]))
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text, split_special_tokens=False):
        del split_special_tokens
        return self.trie_tokenizer.encode(text)[0]

    def _convert_token_to_id(self, token):
        return token

    def _convert_id_to_token(self, index):
        token = self.decoder.get(index, self.unk_token)
        if isinstance(token, bytes):
            token = token.decode("utf-8", errors="replace")
        return token

    def convert_tokens_to_string(self, tokens):
        return b"".join(
            token.encode(errors="replace") if isinstance(token, str) else token
            for token in tokens
        ).decode("utf-8")

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + "rwkv_vocab_v20260603.txt",
            )
        else:
            vocab_file = (
                filename_prefix + "-" if filename_prefix else ""
            ) + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.encoder.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to %s with non-consecutive token ids.",
                        vocab_file,
                    )
                    index = token_index
                writer.write(str(token) + "\n")
                index += 1
        return (vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_ids = [self.bos_token_id] if self.add_bos_token else []
        output = bos_token_ids + token_ids_0
        if token_ids_1 is None:
            return output
        return output + bos_token_ids + token_ids_1

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        if self.add_bos_token:
            prefix = [1]
        else:
            prefix = []

        if token_ids_1 is None:
            return prefix + ([0] * len(token_ids_0))

        return prefix + ([0] * len(token_ids_0)) + prefix + ([0] * len(token_ids_1))

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        bos_token_ids = [self.bos_token_id] if self.add_bos_token else []
        output = len(bos_token_ids + token_ids_0) * [0]
        if token_ids_1 is not None:
            output += len(bos_token_ids + token_ids_1) * [1]
        return output
