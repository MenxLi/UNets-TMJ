# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: trainerThunder.py                                      | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #
import numpy as np
from .trainerVanilla import TrainerVanilla
from monsoonToolBox.misc.progress import printProgressBar

class TrainerThunder(TrainerVanilla):
    def __init__(self, show_progress = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.show_progress = show_progress
        self.save_every = 100
        self.plot_every = 100

        self.save_best = False

    def onTrainStart(self, **kwargs) -> None:
        # self.drawNetStructure(self.save_dir)
        return
	
    def onTrainEpochStart(self, **kwargs) -> None:
        self.lr = self.getLr(self.epoch, self.total_epochs)
        self.setLr(self.lr)
        if self.show_progress:
            printProgressBar(self.epoch, self.total_epochs, prefix=f"{self.epoch}/{self.total_epochs}", length=50)
        return
	
    def onTrainBatchEnd(self, loss, **kwargs) -> None:
        return
	
    def onTrainEpochEnd(self, losses, **kwargs) -> None:
        epoch_history = self.history.setdefault(str(self.epoch), dict())
        train_loss = np.array(losses).mean()
        epoch_history["train_loss"] = train_loss
        epoch_history["lr"] = self.lr
        return 
	
    def onTestEpochEnd(self, test_loss, **kwargs) -> None:
        epoch_history = self.history.setdefault(str(self.epoch), dict())
        epoch_history["test_loss"] = test_loss

		# Save model
        if self.epoch % self.save_every == 0:
            if self.save_dir is not None:
                self.saveModel(self.save_dir, mode = "latest", quiet=True)
		
        if self.save_dir is not None and test_loss<self.min_loss:
            self.min_loss = test_loss
            if self.save_best:
                self.saveModel(self.save_dir, mode="best", quiet=True)

        if self.save_dir is not None and self.epoch % self.plot_every == 0:
            self.drawHistory(self.save_dir)
        return

    def onTrainEnd(self, **kwargs) -> None:
        self.saveModel(self.save_dir, mode="entire")
        if self.show_progress:
            print("Done!")
        return