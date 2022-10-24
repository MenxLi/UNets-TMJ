import argparse, os

import torch
import numpy as np
from progress.bar import Bar
from .trainer.trainerTMJ import TrainerTMJ
from .config import TEMP_DIR
from .dataloading.TMJDataset import TMJDataset2D
import multiprocessing
from monsoonToolBox.misc import readPickle, divideChunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str, default=[], nargs="*", dest="save_dirs")
    parser.add_argument("-b", action="store", default=4, dest="batch_size")
    args = parser.parse_args()

    save_dirs = args.save_dirs
    batch_size = args.batch_size

    test_pickle = os.path.join(TEMP_DIR, "data-test.pkl")
    result_path = os.path.join(TEMP_DIR, "result-upp.npz")

    data_validate = readPickle(test_pickle)
    images = data_validate["imgs"]
    masks = data_validate["msks"]

    device = "cuda"
    trainers = [TrainerTMJ() for _ in range(len(save_dirs))]
    for i in range(len(save_dirs)):
        trainers[i].model = trainers[i].model.to(device)
        trainers[i].loadModel(save_dir=save_dirs[i], mode="best")
        trainers[i].model.eval()

    pred_masks = []
    progress = Bar(max=len(images))
    for imgs, msks in zip(images, masks):
        imgs_batched = list(divideChunks(imgs, batch_size))
        patient_images = imgs
        patient_masks = msks
        patient_masks_pred = None
        patient_masks_preds = []
        for trainer in trainers:
            trainer_pred = []
            for i in range(len(imgs_batched)):
                img_batch = imgs_batched[i]
                img_tensor = torch.tensor(img_batch, dtype=torch.float32).to(device)
                img_tensor.unsqueeze_(1)
                _pred = trainer.model(img_tensor).detach()
                trainer_pred.append(_pred)
            trainer_pred = torch.cat(trainer_pred, dim=0)
            patient_masks_preds.append(trainer_pred.cpu().numpy())
        patient_masks_preds = torch.tensor(patient_masks_preds, device=device)
        patient_masks_pred = patient_masks_preds.mean(dim=0)
        del patient_masks_preds
        patient_masks_pred = patient_masks_pred.argmax(dim=1).cpu().numpy()

        pred_masks.append(patient_masks_pred)
        progress.next()
    np.savez(result_path, imgs=images, masks=pred_masks, labels=masks, dtype=object)
    print("Done!")


if __name__ == "__main__":
    main()
