import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class ComputeLoss:
    def __init__(self):
        self.current_loss = 0
        self.dataset_len = 0

    def add_data(self, value):
        self.current_loss += value
        self.dataset_len += 1

    def get_metric_value(self):
        return self.current_loss / self.dataset_len


def __plotting_metrics(summary):
    fig, axs = plt.subplots(len(summary), squeeze=False, figsize=(15, 15))
    # fig.set_size_inches(10, 20)
    for i, (metric_name, metric_values) in enumerate(summary.items()):
        axs[i][0].set_title(metric_name)
        axs[i][0].plot(metric_values)
    plt.subplots_adjust(hspace=1.0)
    plt.show()


def __prepare_metrics(metrics):
    if metrics is None:
        metrics = {'loss': ComputeLoss}
    else:
        metrics['loss'] = ComputeLoss
    return metrics


def train(dataloader, model, loss_fn, optimizer, epoch_number, validate=False, filename=None, save_every=None, metrics=None):
    save_every = epoch_number if save_every is None else save_every
    metrics = __prepare_metrics(metrics)
    summary = {metric_name : [] for metric_name in metrics.keys()}
    for i in range(epoch_number):
        print(f"\nEpoch {i+1}\n-------------------------------------")
        epoch_metrics = epoch_train(
            dataloader, model, loss_fn, optimizer, metrics)
        for metric_name, metric_calculator in epoch_metrics.items():
            summary[metric_name].append(metric_calculator.get_metric_value())

        if (i+1) % save_every == 0 and filename:
            name_to_save = f'{filename}_{i + 1}'
            torch.save(model.state_dict(), name_to_save)

        if validate:
            epoch_test(dataloader, model, loss_fn)

    __plotting_metrics(summary)
    print("Train Done!")


def epoch_train(dataloader, model, loss_fn, optimizer, metrics):
    data_size = len(dataloader.dataset)

    cont = 0
    last_notified = 0
    metrics_calculators = {metric_name : metric() for metric_name, metric in metrics.items()}
    for X, y in dataloader:
        pred = model(X)
        pred = pred.to(DEVICE)
        y = y.view(-1, 1).squeeze(1)
        y = y.to(DEVICE)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        for metric_name, metric_calculator in metrics_calculators.items():
            if metric_name == 'loss':
                metric_calculator.add_data(loss)
            else:
                metric_calculator.add_data(pred, y)

        cont += len(X)
        if last_notified < cont // 1000:
            last_notified = cont // 1000
            print(f"loss: {loss:>7f} [{cont:>5d}/{data_size:>5d}]")
    return metrics_calculators


def epoch_test(dataloader, model, loss_fn):
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss = loss_fn(pred, y)
