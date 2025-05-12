# data/__init__.py
from .base import BaseDataset, test_batchify_fn
from .ade20k import ADE20KSegmentation

datasets = {
    'ade20k': ADE20KSegmentation,
}

def get_dataset(name, **kwargs):
    key = name.lower()
    if key not in datasets:
        raise ValueError(f"Unknown dataset '{name}'")
    return datasets[key](**kwargs)

def get_available_datasets():
    return list(datasets.keys())
