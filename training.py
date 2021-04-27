import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def __summarize_metric(summaries, metric):
    loss_avg_per_epoch = []
    for epoch_summary in summaries:
        epoch_loss = 0
        for batch_summary in epoch_summary:
            epoch_loss += batch_summary[metric]
        loss_avg_per_epoch.append(epoch_loss/len(epoch_summary))
    return loss_avg_per_epoch

def __plotting_metrics(summary, metrics):
    _, axs = plt.subplots(len(metrics) + 1)
    for i, metric in enumerate(list(metrics.keys()) + ['loss']):
        axs[i].set_title(metric)
        axs[i].plot(__summarize_metric(summary, metric))
    plt.subplots_adjust(hspace=1.0)
    plt.show()



def train(dataloader, model, loss_fn, optimizer, epoch_number, validate=False, filename=None, save_every=None, metrics=None):
    save_every = epoch_number if save_every is None else save_every
    summary = []
    for i in range(epoch_number):
        print(f"\nEpoch {i+1}\n-------------------------------------")
        epoch_summary = epoch_train(dataloader, model, loss_fn, optimizer, metrics)
        summary.append(epoch_summary)

        if (i+1) % save_every == 0 and filename:
            name_to_save = f'{filename}_{i + 1}'
            torch.save(model.state_dict(), name_to_save)


        if validate:
            epoch_test(dataloader, model, loss_fn)
    
    __plotting_metrics(summary, metrics)
    print("Train Done!")


def epoch_train(dataloader, model, loss_fn, optimizer, metrics):
    data_size = len(dataloader.dataset)
    cont = 0
    last_notified = 0
    summary = []
    for X, y in dataloader: 
        batch_summary = {}
        pred = model(X)
        y = y.view(-1,1).squeeze(1)
        y = y.to(DEVICE)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        batch_summary['loss'] = loss
        if metrics:
            for name, metric_function in metrics.items():
                batch_summary[name] = metric_function(pred, y)
        summary.append(batch_summary)

        cont += len(X)
        if last_notified < cont // 1000:
            last_notified = cont // 1000
            print(f"loss: {loss:>7f} [{cont:>5d}/{data_size:>5d}]")
    return summary
        

def epoch_test(dataloader, model, loss_fn):
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss = loss_fn(pred, y)