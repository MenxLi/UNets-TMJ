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
