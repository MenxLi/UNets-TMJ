from PIL.Image import new
from typing import Union, Callable
from numpy import isin
import torch
import warnings
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn
import torch.optim

class TrainerAbstract(object):
	def __init__(self, **kwargs) -> None:
		super().__init__()
		self.epoch = 0
		self.min_loss = 999999

		self.model: torch.nn.Module
		self.optimizer: torch.optim.Optimizer
		self.loss_fn: Callable
		self.batch_size: int
		self.total_epochs: int
		self.base_lr: float

		for k, v in kwargs.items():
			setattr(self, k, v)

		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		
	def onTrainEpochStart(self, **kwargs) -> None:
		return
	
	def onTrainEpochEnd(self, losses = None, **kwargs) -> None:
		return
	
	def onTrainBatchStart(self, **kwargs) -> None:
		return
	
	def onTrainBatchEnd(self, loss = None, **kwargs) -> None:
		return

	def onTestEpochStart(self, **kwargs) -> None:
		return

	def onTestEpochEnd(self, test_loss = None, **kwargs) -> None:
		return

	def onTrainStart(self, **kwargs) -> None:
		return

	def onTrainEnd(self, **kwargs) -> None:
		return

	def checkInitialization(self) -> bool:
		attrs = ["batch_size", "total_epochs", "model", "loss_fn", "optimizer", "base_lr"]
		passed = True
		for attr in attrs:
			if not hasattr(self, attr):
				warnings.warn("{class_name}.{attr_name} is not defined".format(class_name = __class__.__name__, attr_name = attr))
				passed = False
		return passed
	
	def initDevice(self):
		print("Using device: ", self.device)
		self.model = self.model.to(self.device)

	def _checkAttr(self, attr) -> bool:
		if not hasattr(self, attr):
			warnings.warn("{class_name}.{attr_name} is not defined".format(class_name = __class__.__name__, attr_name = attr))
			return False
		return True

	def setLr(self, new_lr: float):
		for g in self.optimizer.param_groups:
			g['lr'] = new_lr
		self.lr = new_lr
	
	def getLr(self, epochs: int, total_epochs: int):
		raise NotImplementedError("getLr() not implemented")
	
	def trainEpochLoop(self, dataloader: DataLoader) -> None:
		size = len(dataloader.dataset)
		self.model.train()
		losses = []

		self.onTrainEpochStart()
		for batch, (X, y) in enumerate(dataloader):
			self.onTrainBatchStart()
			if isinstance(X, tuple) or isinstance(X, list):
				X = [x.to(self.device) for x in X]
			else:
				X = X.to(self.device)
			if isinstance(y, tuple) or isinstance(y, list):
				y = [y_.to(self.device) for y_ in y]
			else:
				y = y.to(self.device)

			# Compute prediction and loss
			pred = self.model(X)
			loss = self.loss_fn(pred, y)

			# Backpropagation
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			losses.append(loss.item())
			self.onTrainBatchEnd(loss = loss.item(), progress = batch*len(X), total_progress = size)
		self.onTrainEpochEnd(losses = losses)

	def testEpochLoop(self, dataloader: DataLoader) -> None:
		size = len(dataloader.dataset)
		num_batches = len(dataloader)
		test_loss = 0
		self.model.eval()

		self.onTestEpochStart()
		with torch.no_grad():
			for X, y in dataloader:
				if isinstance(X, tuple) or isinstance(X, list):
					X = [x.to(self.device) for x in X]
				else:
					X = X.to(self.device)
				if isinstance(y, tuple) or isinstance(y, list):
					y = [y_.to(self.device) for y_ in y]
				else:
					y = y.to(self.device)

				pred = self.model(X)
				test_loss += self.loss_fn(pred, y).item()

		test_loss /= num_batches
		self.onTestEpochEnd(test_loss= test_loss)
	
	def _train(self, train_dataloader: DataLoader, test_dataloader: Union[DataLoader, None] = None) -> torch.nn.Module:
		if not self.checkInitialization():
			print("Falied to start the training process")
			return
		self.initDevice()
		self.onTrainStart()
		while(self.epoch < self.total_epochs):
			self.trainEpochLoop(train_dataloader)
			if not test_dataloader is None:
				self.testEpochLoop(test_dataloader)
			self.epoch += 1
		self.onTrainEnd()
		return self.model
	
	def feed(self, train_dataset: Dataset, test_dataset: Union[Dataset, None], **dataloader_params):
		assert isinstance(train_dataset, Dataset), "train_dataset has to be subclass of torch.utils.data.Dataset"
		self.train_dataset = train_dataset
		self.test_dataset = test_dataset
		self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, **dataloader_params)
		if test_dataset is None:
			self.test_dataloader = None
		else:
			self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, **dataloader_params)
	
	def train(self):
		"""
		Train the model, datasets should be feed with <trainer>.feed(*params) in advance
		"""
		assert hasattr(self, "train_dataloader"), "Dataset not feed, use {}.feed(<params>) to feed dataset.".format(__class__.__name__)
		self._train(self.train_dataloader, self.test_dataloader)


	


