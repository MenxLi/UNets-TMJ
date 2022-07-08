# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: divideIter.py                                          | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #
import typing

def divideChunks(lis: typing.Iterable, n_size: int) -> typing.Iterator:
    for i in range(0, len(lis), n_size): 
        yield lis[i:i + n_size]

def divideFold(lis: list, fold:int, total_fold: int) -> typing.Tuple[typing.Iterable, typing.Iterable]:
    chunk_size = len(lis)//total_fold
    split = divideChunks(lis, chunk_size)
    split = list(split)
    a = split.pop(fold)
    b = []
    for i in split:
        b += i
    return a, b
