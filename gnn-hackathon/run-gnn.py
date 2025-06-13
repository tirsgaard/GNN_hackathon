from src.herg_datamodule import HERGDataModule
from src.gnn_models import SimpleGraphConv, MessagePassingGNN, EnsembleConnectedGNN, Normalizer, OutputNormalizer
from src.submit import send_result
from src.train import train, validate
from src.test_func import test

import torch

torch.manual_seed(42)  # For reproducibility

# Name for model to be submitted
model_name = "Simple GNN"
author = "rhti"
# Limit torch to 4 threads
torch.set_num_threads(4)

# Hyperparameters
n_rounds = 2
filter_length = 64
learning_rate = 0.01
epochs = 1000
dropout = 0.4
submit_test = False

# The ADMET group benchmarks
results = {}
task_name = "hERG_at_10uM_small"  # Example task name, can be changed to any other task

# Setup data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
dataset.prepare_data()
dataset.setup()
train_dataloader = dataset.train_dataloader()
val_dataloader = dataset.val_dataloader()    
batch = next(iter(train_dataloader))


# Initialize the model
loss_function = torch.nn.MSELoss(reduction='none')
n_features = batch.x.shape[-1] if batch.x is not None else 0
print(n_features)
normalizer = Normalizer(torch.zeros(n_features), torch.ones(n_features))
normalizer.estimate(dataset.train_dataset)
model = MessagePassingGNN(n_features, filter_length, n_rounds, normalizer, is_classification=False, dropout=dropout).to(device)
if device.type == 'cuda':
    model = torch.compile(model, dynamic=True, fullgraph=True)
target_transform = OutputNormalizer(torch.zeros(1), torch.ones(1), False).to(device)
target_transform.estimate(dataset.train_dataset)

# Train the model
model = train(model, train_dataloader, val_dataloader, loss_function, target_transform, device, epochs=epochs, learning_rate=learning_rate, n_val_epoch=5)
# Evaluate the model
val_error, var_corr = validate(model, val_dataloader, loss_function, target_transform, device)
print(f"Validation MSE: {val_error}, Validation Variance Correlation: {var_corr}")

# Make predictions on the test set
error = test(model, target_transform, dataset, device)

results = {}
if submit_test:
# Perform tests
    predictions = {}
    predictions[task_name] = error
    result = {task_name: {'mae': error}}
    
    print(f"Test results for {task_name}: {result}")
    results[task_name] = result

if submit_test:
    # Format of the output
    output = {"results": results,
            "model_name": model_name,
            "author": author,
            "extra_data": {}}
    send_result(output)
