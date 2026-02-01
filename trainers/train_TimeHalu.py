import torch
import torch.nn as nn
import os
import json
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


default_scheduler_args = {
    'mode': 'max', 
    'factor': 0.1, 
    'patience': 5,
    'threshold': 0.00001, 
    'threshold_mode': 'rel',
    'cooldown': 0, 
    'min_lr': 1e-8, 
    'eps': 1e-08, 
    'verbose': True
}


def train_TimeHalu(
    model,
    train_loader,
    val_tuple,
    test_tuple,
    num_epochs,
    save_path,
    optimizer,
    scheduler_args = default_scheduler_args,
    early_stopping = True,
    show_sizes = False,
    use_scheduler = False,
    scheduler = None,
    patience = 10
):
    '''
    Train TimeHalu model
    '''
    best_val_metric = -1e9
    wait_for_scheduler = 20
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_args)

    counter = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_pred_loss = 0.0
        total_mask_loss = 0.0
        for i, (X, times, Y) in enumerate(train_loader):
            X = X.transpose(0, 1)
            times = times.transpose(0, 1)
            optimizer.zero_grad()
            out = model(X, times, Y)

            loss = out['loss']
            pred_loss = out['pred_loss']
            mask_loss = out['mask_loss']

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_pred_loss += pred_loss.item()
            total_mask_loss += mask_loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_pred_loss = total_pred_loss / len(train_loader)
        avg_mask_loss = total_mask_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Pred Loss: {avg_pred_loss:.4f}, Mask Loss: {avg_mask_loss:.4f}")


        # Validation
        model.eval()
        with torch.no_grad():
            X, times, y = val_tuple
            out = model(X, times, y, captum_input=False)
            pred = out['y_pred_masked']

            y_true = y.cpu().numpy()
            y_prob = pred.detach().cpu().numpy()
            val_auc = roc_auc_score(y_true, y_prob[:,1], multi_class='ovr')

            X, times, y = test_tuple
            out = model(X, times, y, captum_input=False)
            pred = out['y_pred_masked']

            y_true = y.cpu().numpy()
            y_prob = pred.detach().cpu().numpy()
            test_auc = roc_auc_score(y_true, y_prob[:,1])
        print(f'Val AUC: {val_auc:.4f}')
        print('Test AUROC: {:.4f}'.format(test_auc))

        # ---------------- Early Stopping ----------------
        met = val_auc   
        cond = not early_stopping
        if early_stopping:
            cond = (met > best_val_metric)

        if cond:
            best_val_metric = met
            if save_path is not None:
                torch.save(model.state_dict(), save_path)   
            best_epoch = epoch
            counter = 0  # reset counter，因为有提升
            print('Save at epoch {}: Val AUC={:.4f}'.format(epoch, met))
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch} with Val AUC={best_val_metric:.4f}")
                break

        # ---------------- Scheduler ----------------
        if use_scheduler and (epoch > wait_for_scheduler):
            scheduler.step(met)

    print("Training complete.")
    return model
    