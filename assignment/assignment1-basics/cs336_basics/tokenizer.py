import json
import regex as re
from tests.common import gpt2_bytes_to_unicode
from collections.abc import Iterable, Iterator


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = []
        if special_tokens:
            # 如果 special_tokens 存在包含关系，被包含的需要在包含他的后面，否则会识别错误
            for token in special_tokens:
                t = 0
                for other in self.special_tokens:
                    if token in other:
                        self.special_tokens = self.special_tokens + [token]
                        t = 1
                        break
                if t == 0:
                    self.special_tokens = [token] + self.special_tokens
        self.vocab_rev = {}
        for i in vocab.keys():
            self.vocab_rev[vocab[i]] = i
        # 为了优化 merge 逻辑，把 merge 从列表变成字典，让查询速度变成 O(1)
        # 同时，这里 merge 里存的值是优先级，用于后续合并判断
        self.merges_dict = {}
        i = 0
        for merge in self.merges:
            self.merges_dict[merge] = i
            i += 1

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # 获取 byte的一部分(表现形式为 int) 到 gpt2 编码的映射以及其逆映射
        gpt2_dict = gpt2_bytes_to_unicode()
        gpt2_dict_rev = {}
        for byte in gpt2_dict.keys():
            gpt2_dict_rev[gpt2_dict[byte]] = byte
        # 进行 vocab 的文件处理
        vocab = {}
        with open(vocab_filepath, encoding="utf-8") as f:
            # json 文件默认得到 GPT-2 编码到 token ID 的映射
            # 而我们需要的 vocab,是 token ID 到原始 byte 的映射
            vocab_json = json.load(f)
        # 对于每一个 vocab 中的 token, 进行遍历并转化
        for token in vocab_json.items():
            vocab_list = []
            token_gpt = token[0]
            token_id = token[1]
            for char in token_gpt:
                vocab_list.append(gpt2_dict_rev[char])
            vocab_byte = bytes(vocab_list)
            vocab[token_id] = vocab_byte
        # 进行 merge 的文件处理
        merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            # 除掉第一行 version
            next(f, None)
            for line in f:
                line = line.rstrip()
                left, right = line.split(" ")
                left_list, right_list = [], []
                # 从 gpt2 的映射中得到的是整数，而非 byte, 所以还要经过一次转化
                for char in left:
                    byte = gpt2_dict_rev[char]
                    left_list.append(byte)
                for char in right:
                    byte = gpt2_dict_rev[char]
                    right_list.append(byte)
                left_byte = bytes(left_list)
                right_byte = bytes(right_list)
                merges.append((left_byte, right_byte))
        return cls(vocab, merges, special_tokens)

    # 将文章编码成分词后的字节序列
    def _resolve_encode_logic(self, text: str) -> list[int]:
        if self.special_tokens:
            special_pattern = "|".join(re.escape(t) for t in self.special_tokens)
            special_pattern = "(" + special_pattern + ")"
        else:
            special_pattern = None
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # 按照特殊字符分割(保留特殊字符)
        if special_pattern:
            text_splitted = re.split(special_pattern, text)
        else:
            text_splitted = [text]
        # 按照正则标准分割
        text_encoded = []

        for piece in text_splitted:
            # 跳过空内容
            if not piece:
                continue
            # 特殊字符存入text中
            if piece in self.special_tokens:
                piece_byte = piece.encode("utf-8")
                text_encoded.append(self.vocab_rev[piece_byte])
                continue
            for m in re.finditer(PAT, piece):
                pre_token = m.group().encode("utf-8")
                token_tuple = tuple(pre_token[i : i + 1] for i in range(len(pre_token)))
                # 对 pair 查询，而不是对 merges 查询，大大提高速度
                while 1:
                    maxi = 1e9
                    maxpos = -2                  
                    for i in range(len(token_tuple) - 1):
                        pair = (token_tuple[i], token_tuple[i + 1])
                        if pair in self.merges_dict and self.merges_dict[pair] < maxi:
                            maxi = self.merges_dict[pair]
                            maxpos = i
                    if maxpos == -2:
                        break
                    token_tuple = self._helper_merge(token_tuple, maxpos)
                # 按照词表来查表，把 byte 列表转换到 token ID
                for byte in token_tuple:
                    text_encoded.append(self.vocab_rev[byte])
        return text_encoded

    def encode(self, text: str) -> list[int]:
        return self._resolve_encode_logic(text)
        
    # 负责应用一个规则到 tuple 上
    @staticmethod
    def _helper_merge(
        token_tuple: tuple[bytes, ...], pos: int
    ) -> tuple[bytes, ...]:
        new_word = []
        i = 0
        # 使用 while 而非 for，处理跳过元素的情况
        while i < len(token_tuple) - 1:
            if i == pos:
                new_word.append(token_tuple[i] + token_tuple[i + 1])
                i += 2
            else:
                new_word.append(token_tuple[i])
                i += 1
        if i < len(token_tuple):
            new_word.append(token_tuple[i])
        new_word_tuple = tuple(new_word)
        return new_word_tuple

    def encode_iterable(self, text: Iterable[str]) -> Iterator[int]:
        for chunk in text:
            text_encoded = self._resolve_encode_logic(chunk)
            yield from text_encoded

    def decode(self, text: list[int]) -> str:
        text_decoded = b""
        for byte in text:
            text_decoded += self.vocab[byte]
        text_str = text_decoded.decode("utf-8", errors="replace")
        return text_str
