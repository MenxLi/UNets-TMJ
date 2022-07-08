from .nnUNetTrainerV2 import nnUNetTrainerV2

class TrainerCustomV2(nnUNetTrainerV2):
	def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
		super().__init__(plans_file, fold, output_folder=output_folder, dataset_directory=dataset_directory, batch_dice=batch_dice, stage=stage, unpack_data=unpack_data, deterministic=deterministic, fp16=fp16)
		self.max_num_epochs=300
