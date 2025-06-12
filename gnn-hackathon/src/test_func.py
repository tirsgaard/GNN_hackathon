import torch

def test(model, target_transform, dataset, device):
    dataset.setup('test')
    y_pred = []
    y_target = []
    for batch in dataset.test_dataloader():
        with torch.no_grad():
            out = model(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device), batch_size=batch.target.shape[0])
            out_transformed = target_transform.unnormalize(out)
            y_pred.append(out_transformed.cpu())
            y_target.append(batch.target.to(torch.float32).cpu())
    y_target = torch.hstack(y_target).numpy()
    y_pred = torch.hstack(y_pred).numpy()
    mean_error = ((y_pred - y_target) ** 2).mean()
    return mean_error.item()