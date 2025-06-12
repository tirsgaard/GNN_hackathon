import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train(model, train_dataloader, val_dataloader, loss_function, target_transform, device, epochs=10, learning_rate=0.01, n_val_epoch=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3) # added ReduceLROnPlateau
    for epoch in range(epochs):
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            out = model(batch.x.to(device), batch.edge_index.to(device), batch=batch.batch.to(device), batch_size=batch.target.shape[0])
            y_target = target_transform(batch.target.to(torch.float32).to(device))
            loss = loss_function(out, y_target).mean()
            loss.backward()
            optimizer.step()
        if epoch % n_val_epoch == 0:
            val_error, var_corr = validate(model, val_dataloader, loss_function, target_transform, device)
            print(f"Epoch {epoch}, Validation error: {val_error}, Validation Variance Correlation: {var_corr}, Train loss: {loss.item()}")
            scheduler.step(val_error) # added scheduler step
            # Print if the learning rate is being adjusted
            if optimizer.param_groups[0]['lr'] < learning_rate:
                print(f"Learning rate adjusted to {optimizer.param_groups[0]['lr']}")
                learning_rate = optimizer.param_groups[0]['lr']
            
            # Break if learning rate is too low
            if optimizer.param_groups[0]['lr'] < 1e-6:
                print("Learning rate too low, stopping training.")
                break
    return model


def validate(model, val_dataloader, loss_function, target_transform, device):
    model.eval()
    errors = []
    vars = []
    for batch in val_dataloader:
        with torch.no_grad():
            out, var = model(batch.x.to(device), batch.edge_index.to(device), batch=batch.batch.to(device), batch_size=batch.target.shape[0], return_var=True)
            out_transformed = target_transform.unnormalize(out)
            var_transformed = target_transform.unnormalize(var)
            error = loss_function(out_transformed, batch.target.to(torch.float32).to(device))
            errors.append(error.cpu())
            vars.append(var_transformed.cpu())
    errors = torch.hstack(errors)
    vars = torch.hstack(vars)
    return errors.mean().item(), torch.corrcoef(torch.stack([errors, vars]))[0, 1].item()