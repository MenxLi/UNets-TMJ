import numpy as np
import torch
import torch.optim
from .trainerAbstract import TrainerAbstract
from ..learningRate import PolyLR
import time, os, json
import matplotlib.pyplot as plt

pJoin = os.path.join


class TrainerVanilla(TrainerAbstract):
    WEIGHTS_LATEST_FNAME = "weights_latest.pth"
    WEIGHTS_BEST_FNAME = "weights_best.pth"
    HISTORY_FNAME = "history.json"
    STATUS_FNAME = "status.json"

    def __init__(self, **kwargs) -> None:

        self.history = dict()
        self.save_every = 1
        self.save_dir = None
        super().__init__(**kwargs)

    def getLr(self, epochs: int, total_epochs: int):
        if not hasattr(self, "lr_instance"):
            lr_instance = PolyLR(power=2)
            self.lr_instance = lr_instance
            print("Using default poly lr: ", str(self.lr_instance))
        return self.lr_instance(epochs, total_epochs, self.base_lr)

    def saveModel(self, save_dir, mode="latest"):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # Save weights
        if mode == "latest":
            w_fname = self.WEIGHTS_LATEST_FNAME
        elif mode == "best":
            w_fname = self.WEIGHTS_BEST_FNAME
        else:
            raise Exception("Saving mode can be either latest or best.")
        torch.save(self.model.state_dict(), pJoin(save_dir, w_fname))

        # History dict
        with open(pJoin(save_dir, self.HISTORY_FNAME), "w") as fp:
            json.dump(self.history, fp)

        # Status dict
        status = {
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            # "base_lr": self.base_lr,
            "min_loss": self.min_loss,
        }
        with open(pJoin(save_dir, self.STATUS_FNAME), "w") as fp:
            json.dump(status, fp)

        # Hyper-parameters
        hyper_param = {
            "base_lr": str(self.base_lr),
            "optimizer": str(self.optimizer),
            "loss_fn": str(self.loss_fn),
        }
        if hasattr(self, "batch_size"):
            hyper_param["batch_size"] = str(self.batch_size)
        if hasattr(self, "lr_instance"):
            hyper_param["lr_instance"] = str(self.lr_instance)
        comment = ""
        for k, v in hyper_param.items():
            comment += "{} - {}\n".format(k, v)
        with open(pJoin(save_dir, "comment.txt"), "w") as fp:
            fp.write(comment)

        print("Model ({}) saved.".format(mode))

    def loadModel(self, save_dir, mode="latest"):
        # Load weights
        if mode == "latest":
            w_fname = self.WEIGHTS_LATEST_FNAME
        elif mode == "best":
            w_fname = self.WEIGHTS_BEST_FNAME
        else:
            raise Exception("Saving mode can be either latest or best.")
        self.model.load_state_dict(torch.load(pJoin(save_dir, w_fname)))
        print("loaded the model weights - {}".format(mode))

        # Load history
        with open(pJoin(save_dir, self.HISTORY_FNAME), "r") as fp:
            self.history = json.load(fp)

        # Load status
        with open(pJoin(save_dir, self.STATUS_FNAME), "r") as fp:
            status = json.load(fp)
        for k, v in status.items():
            setattr(self, k, v)

    def drawHistory(self, save_dir):
        his_len = len(self.history)
        train_loss = [self.history[str(i)]["train_loss"] for i in range(his_len)]
        test_loss = [self.history[str(i)]["test_loss"] for i in range(his_len)]
        lr = [self.history[str(i)]["lr"] for i in range(his_len)]

        lr = np.array(lr)
        if lr.max() == 0:
            lr_scale = 1
        else:
            lr_scale = 1 / (lr.max())

        fig, ax = plt.subplots()
        ax.plot(train_loss, label="train-loss", color="red")
        ax.plot(test_loss, label="test-loss", color="blue")
        ax.plot(lr * lr_scale, label="lr*{:.2f}".format(lr_scale), color="green")
        ax.legend(loc='upper right')
        ax.set_xlabel("Epochs")
        plt.savefig(pJoin(save_dir, "progress.png"))
        del fig, ax

    ##=================================CallBacks====================================

    def onTrainStart(self, **kwargs) -> None:
        self.timer = time.time()
        self.epoch_timer = time.time()
        return super().onTrainStart()

    def onTrainEpochStart(self, **kwargs) -> None:
        self.epoch_timer = time.time()
        print(f"\nEpoch {self.epoch}\n-------------------------------")
        self.lr = self.getLr(self.epoch, self.total_epochs)
        self.setLr(self.lr)
        print("Learning rate: ", self.lr)
        return super().onTrainEpochStart()

    def onTrainBatchEnd(self, loss, **kwargs) -> None:
        total_progress = kwargs["total_progress"]
        progress = kwargs["progress"]
        if progress:
            print(f"loss: {loss:>7f}  [{progress:>5d}/{total_progress:>5d}]", end="\r")
        return super().onTrainBatchEnd(loss=loss, **kwargs)

    def onTrainEpochEnd(self, losses, **kwargs) -> None:
        epoch_history = self.history.setdefault(str(self.epoch), dict())
        train_loss = np.array(losses).mean()
        epoch_history["train_loss"] = train_loss
        epoch_history["lr"] = self.lr
        print("\r" + " " * 30)  # clear output
        print(f"Train loss: {train_loss:>8f}")

        return super().onTrainEpochEnd(losses=losses)

    def onTestEpochEnd(self, test_loss, **kwargs) -> None:
        print(f"Test loss: {test_loss:>8f}")
        print("This epoch took: {}s".format(np.round(time.time() - self.epoch_timer)))

        epoch_history = self.history.setdefault(str(self.epoch), dict())
        epoch_history["test_loss"] = test_loss

        # Save model
        if self.epoch % self.save_every == 0:
            if self.save_dir is not None:
                self.saveModel(self.save_dir, mode="latest")

        if self.save_dir is not None and test_loss < self.min_loss:
            self.min_loss = test_loss
            self.saveModel(self.save_dir, mode="best")

        if self.save_dir is not None:
            self.drawHistory(self.save_dir)

        return super().onTestEpochEnd(test_loss=test_loss)

    def onTrainEnd(self, **kwargs) -> None:
        print("Done!")
        print("Training took: {}s".format(time.time() - self.timer))
        return super().onTrainEnd()
