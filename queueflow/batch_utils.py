from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch_geometric


def batch_to_numpy_dict(batch):
    def tonumpy(element):
        if torch.is_tensor(element):
            return element.numpy()
        elif isinstance(element, list):
            return [tonumpy(ee) for ee in element]
        elif isinstance(element, dict):
            return {k: tonumpy(ee) for k, ee in element.items()}
        elif element is None:
            return None
        elif isinstance(element, (int, str, float)):
            return element
        else:
            raise ValueError

    batch_new = torch_geometric.data.Batch().from_dict(
        {k: tonumpy(v) for k, v in batch.to_dict().items()}
    )
    return batch_new
