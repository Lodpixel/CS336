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
                
    def encode(self, text) -> list[int]:
        special_pattern = "|".join(re.escape(t) for t in self.special_tokens)
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # 按照特殊字符分割
        if (special_pattern):
            text_splitted = re.split(special_pattern, text)
        else:
            text_splitted = [text]
        word_list = {}
        # 按照正则标准分割
        for piece in text_splitted:
            if not piece:
                continue
            for m in re.finditer(PAT, piece):
                pre_token = m.group().encode("utf-8")
                token_tuple = tuple(pre_token[i:i+1] for i in range(len(pre_token)))
                # 重复的词直接在这里用字典计数，后续不需要反复计算和合并 
                word_list[token_tuple] = word_list.get(token_tuple, 0) + 1
        vocab = {}