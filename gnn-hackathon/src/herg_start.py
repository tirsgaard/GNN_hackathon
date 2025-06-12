from herg_datamodule import HERGDataModule

task_name = "hERG_at_10uM_small"  # Example task name, can be changed to any other task
dataset = HERGDataModule(task_name,
        data_dir = "./data",
        batch_size = 1024,
        seed = 42,
        split_type = "scaffold",
        val_fraction = 0.1,
        with_hydrogen = False,
        kekulize = False,
        with_descriptors = False,
        with_fingerprints = False,
        only_use_atom_number=True,
    )
dataset.setup()
train_dataloader = dataset.train_dataloader()
val_dataloader = dataset.val_dataloader()    
test_dataloader = dataset.test_dataloader()


batch = next(iter(train_dataloader))
test = 2