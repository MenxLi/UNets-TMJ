import re
import typing, math, os, io
import numpy as np
import multiprocessing, tempfile, pickle
from .divideIter import divideChunks

def lisJobParallel(func: typing.Callable, list_like: typing.Iterable, use_buffer:bool = True, n_workers: int = -1) -> list:
	"""
	Parallel a job that is applied to a list-like object
	The paralleled function should only take one argument which is the list_like
	the function should be able to be run on subset of the list_like
	and return list-like results or None
	i.e. 
		func(list_like) -> list_like2 | None
	- list_like: list-like argument
	- n_workers: number of process
	- use_buffer: use hard disk as buffer for each subprocess output, enable when the data exchange is large
		if the program stuck at Pipe connection, try set this to True
	"""
	if n_workers <= 0:
		n_workers = inferWorkers()

	data = list_like
	chunk_size = math.ceil(len(data)/n_workers)
	conns = []
	procs = []

	# define the function for multiprocessing.Process
	def processFunc(conn, func_, data_):
		out_ = func_(data_)
		if use_buffer:
			f_path = tempfile.NamedTemporaryFile(mode = "w+b").name
			with open(f_path, "wb") as fp:
				bytes = pickle.dumps(out_)
				fp.write(bytes)
			conn.send(f_path)
		else:
			conn.send(out_)

	# Create and run the processes	
	for d in divideChunks(data, chunk_size):
		conn1, conn2 = multiprocessing.Pipe()
		process = multiprocessing.Process(\
			target=processFunc, args=(conn1, func, d))
		conns.append(conn2)
		procs.append(process)
		process.start()
	for p in procs:
		p.join()
	
	# Concatenate results
	out = []
	for conn_ in conns:
		obj = conn_.recv()
		if use_buffer:
			with open(obj, "rb") as fp:
				out_ = pickle.loads(fp.read())
			os.remove(obj)	# delete the temporary buffer file
		else:
			out_ = obj
		# concatenate
		if out_ is not None:
			out = [*out, *out_]
		else:
			out.append(None)
	return out

def mapJobParallel(func: typing.Callable, lis: typing.Iterable, **kwargs) -> list:
	"""
	Parallel a job that is applied to each element of the lis
	The paralleled function should only take one argument which is any element of the lis
	the function should be able to be run on each element of the lis
	and return a list of all of the results
	i.e. 
		func(lis_elem) -> retuen_elem
	- lis: list-like argument
	- n_workers: number of process
	- use_buffer: use hard disk as buffer for each subprocess output, enable when the data exchange is large
		if the program stuck at Pipe connection, try set this to True
	"""
	def _func(_lis):
		return [func(elem) for elem in _lis]
	return lisJobParallel(func=_func, list_like=lis, **kwargs)

def inferWorkers(relax = 1) -> int:
	workers = multiprocessing.cpu_count() - relax
	return workers


def lisJobParallel_(func: typing.Callable, list_like: typing.Iterable, n_workers: int = -1) -> list:
	"""
	Parallel a job that is applied to a list-like object (Use multiprocessing.Pool)
	The paralleled function should only take one argument which is the list_like
	the function should be able to be run on subset of the list_like
	and return list-like results or None
	i.e. 
		func(list_like) -> list_like2 | None
	- list_like: list-like argument
	- n_workers: number of process
	"""
	if n_workers <= 0:
		n_workers = inferWorkers()

	data = list_like
	chunk_size = math.ceil(len(data)/n_workers)

	pool = multiprocessing.Pool(n_workers)
	results = pool.map(func, divideChunks(data, chunk_size))
	out = []
	for out_ in results:
		if out_ is not None:
			out = [*out, *out_]
		else:
			out.append(None)
	return out