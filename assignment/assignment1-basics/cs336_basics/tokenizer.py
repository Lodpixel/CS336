import json
import regex as re


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, encoding="utf-8") as f:
            vocab = json.load(f)
        merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                merges.append(line)
        return cls(vocab, merges, special_tokens)

    # 将文章编码成分词后的字节序列
    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            special_pattern = "|".join(re.escape(t) for t in self.special_tokens)
        else:
            special_pattern = None
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # 按照特殊字符分割
        if special_pattern:
            text_splitted = re.split(special_pattern, text)
        else:
            text_splitted = [text]
        # 按照正则标准分割
        text_encoded = []
        for piece in text_splitted:
            if not piece:
                continue
            for m in re.finditer(PAT, piece):
                pre_token = m.group().encode("utf-8")
                token_tuple = tuple(pre_token[i : i + 1] for i in range(len(pre_token)))
                # 对于每一个分出来的词，按顺序应用每一条规则进行合并
                for rule in self.merges:
                    token_tuple = self._helper_merge(token_tuple, rule)
                # 按照词表来查表，把 byte 列表转换到 token ID
                for byte in token_tuple:
                    text_encoded.append(self.vocab(byte))
        
        
        
        
                

    # 负责应用一个规则到 tuple 上
    def _helper_merge(token_tuple: tuple[bytes, ...], mcomb: int) -> tuple[bytes, ...]:
        new_list = {}
        new_word = []
        i = 0
        # 使用 while 而非 for，处理跳过元素的情况
        while i < len(token_tuple) - 1:
            if token_tuple[i] == mcomb[0] and token_tuple[i + 1] == mcomb[1]:
                new_word.append(mcomb[0] + mcomb[1])
                i += 2
            else:
                new_word.append(token_tuple[i])
                i += 1
        if i < len(token_tuple):
            new_word.append(token_tuple[i])
        new_word_tuple = tuple(new_word)
        return new_word_tuple

    def encode_iterable(self):
        return

    def decode(self, text: list[int]) -> str:
        return
