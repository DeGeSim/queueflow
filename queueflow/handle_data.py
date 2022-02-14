from types import GeneratorType

import torch
import torch_geometric


class HandleDataBase:
    def _clone_tensors(self, wkin):
        if isinstance(wkin, list):
            return [self._clone_tensors(e) for e in wkin]
        elif isinstance(wkin, GeneratorType):
            return (self._clone_tensors(e) for e in wkin)
        elif isinstance(wkin, torch_geometric.data.batch.Data):
            return self.clone_batch(wkin)
        elif isinstance(wkin, torch.Tensor):
            return wkin.clone()
        return wkin

    def move(self, element, device):
        """This function moves batches (eg. from torch_geometric) to a specified device
        and also takes into account manually assinged properties."""

        if isinstance(element, torch_geometric.data.Data):
            element_new = element.to(device)
        #     element_new = torch_geometric.data.Batch().from_dict(
        #         {k: move(v, device) for k, v in element.to_dict().items()}
        #     )
        #     for attr in [
        #         "__slices__",
        #         "__cat_dims__",
        #         "__cumsum__",
        #         "__num_nodes_list__",
        #         "__num_graphs__",
        #     ]:
        #         if hasattr(element_new, attr):
        #             setattr(element_new, attr, move(getattr(element, attr), device))
        elif hasattr(element, "to") and callable(getattr(element, "to")):
            element_new = element.to(device)
        elif isinstance(element, (List, Set, Tuple)):
            element_new = type(element)((move(ee, device) for ee in element))
        elif isinstance(element, Dict):
            element_new = {k: move(v, device) for k, v in element.items()}
        elif isinstance(element, (int, str, float)):
            element_new = element
        elif type(element).__module__ == np.__name__:
            element_new = element
        elif element is None:
            element_new = None
        else:
            raise RuntimeError(
                "Cannot move this object to the torch device, invalid type."
            )
        del element
        return element_new

    def clone_or_copy(self, element):
        if isinstance(element, torch_geometric.data.Data):
            return self.clone_batch(device)
        if hasattr(element, "clone") and callable(getattr(element, "clone")):
            return element.clone()
        elif isinstance(element, (List, Set, Tuple)):
            return type(element)((clone_or_copy(ee) for ee in element))
        elif isinstance(element, Dict):
            return {k: clone_or_copy(ee) for k, ee in element.items()}
        elif isinstance(element, (int, str, float)):
            return element
        elif type(element).__module__ == np.__name__:
            return element
        elif element is None:
            return None
        else:
            raise ValueError

    def clone_batch(self, batch):
        """This function clones batches (eg. from torch_geometric) and
        also takes into account manually assinged properties. This is needed
        when using torch_geometric with torch.multiprocessing"""
        # batch_cloned = torch_geometric.data.Batch().from_dict(
        #     {k: clone_or_copy(v) for k, v in batch.to_dict().items()}
        # )

        # for attr in [
        #     "__slices__",
        #     "__cat_dims__",
        #     "__cumsum__",
        #     "__num_nodes_list__",
        # ]:
        #     if hasattr(batch, attr):
        #         setattr(batch_cloned, attr, clone_or_copy(getattr(batch, attr)))
        batch_cloned = batch.clone()
        return batch_cloned
