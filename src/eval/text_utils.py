from typing import List


def ngram_repeat_rate(ids: List[int], n: int = 2) -> float:
    """n-gram反復率: ユニークn-gramの割合の逆（1 - unique/total）。
    ids: token id 列
    """
    if len(ids) < n:
        return 0.0
    total = max(1, len(ids) - n + 1)
    grams = [tuple(ids[i:i+n]) for i in range(total)]
    uniq = len(set(grams))
    return 1.0 - (uniq / total)
