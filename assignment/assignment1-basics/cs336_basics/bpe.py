import regex as re
    
# 初始化词表
def vocab_init(
    vocab: dict[int, bytes],
    special_tokens: list[str]
):
    for i in range(256):
        vocab[i] = bytes([i])
    for i in range(len(special_tokens)):
        vocab[256 + i] = special_tokens[i].encode("utf-8")

# 找到需要合并的位置并合并
def merge(
    word_list: dict[tuple[bytes, ...], int],
    vocab: dict[int, bytes],
    vocab_cur_size: int,
    merges: list[tuple[bytes, bytes]]
) -> dict[tuple[bytes, ...], int]:
    count = {}
    # 找最多的相邻 vocab
    for word in word_list.keys():
        for i in range(len(word) - 1):
            count[(word[i], word[i + 1])] = count.get((word[i], word[i + 1]), 0) + word_list[word]
    mcomb = max(count.items(), key=lambda kv: (kv[1], kv[0]))[0]
    # 存入词表和 merge 记录
    merges.append((mcomb[0], mcomb[1]))
    vocab[vocab_cur_size] = mcomb[0] + mcomb[1]
    vocab_cur_size += 1
    # 更新 word_list（合并）
    new_list = {}
    for word in word_list.keys():
        new_word = []
        i = 0
        # 使用 while 而非 for，处理跳过元素的情况
        while i < len(word) - 1:
            if (word[i] == mcomb[0] and word[i + 1] == mcomb[1]):
                new_word.append(mcomb[0] + mcomb[1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        if (i < len(word)):
            new_word.append(word[i])
        new_word_tuple = tuple(new_word)
        new_list[new_word_tuple] = new_list.get(new_word_tuple, 0) + word_list[word]
    return new_list


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, encoding="utf-8") as f:
        text = f.read()
    special_pattern = "|".join(re.escape(t) for t in special_tokens)
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
            # 重复的词直接在这里用字典计数，后续不需要反复计算和合并 j
            word_list[token_tuple] = word_list.get(token_tuple, 0) + 1
    vocab = {}
    merges = []
    vocab_init(vocab, special_tokens)
    vocab_cur_size = len(vocab)
    # 反复进行 merge 操作
    while(vocab_cur_size < vocab_size):
        word_list = merge(word_list, vocab, vocab_cur_size, merges)
        vocab_cur_size += 1
    return (vocab, merges)