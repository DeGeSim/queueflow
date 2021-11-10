# queueflow
This is a library for parallel processing using the `torch.multiprocessing` library.
Its goal is to bring the data as fast as possible to the GPU, even if it requires
heave preprocessing and exceeds the size of the RAM. 

This is the general idea:
```python
pseq = Sequence(
        ProcessStep(read_chunk, 2),
        PoolStep(process_chunk,nworkers=20),
        RepackStep(batch_size),
        ProcessStep(torch_geometric.data.Batch().from_data_list),
        Queue(prefetch_batches)
...
pseq.queue_iterable(epoch_chunks)
...
for batch in pseq(filelist):
    prediction = model(batch)
```

Classes:
[Sequence] Outward facing class, that manages the processes, queues and pools.
The input is a sequence of other classes as well as `multiprocessing.Queue`.
If there is no Queue between two elements, a Queue holding at most one element 
is automatically inserted. For the example above, this yield the following graph.
`read_chunk -- Queue(1) -- process_chunk -- Queue(1) -- RepackStep(batch_size) -- Queue(prefetch_batches)`

[Process] A set of workers applies the given function to each of the element, asynchronously.
[Pool] A pool of workers applies the given function to each of the yielded elements.
The output will be a List. This is frequently preferable for large sets of small tensors,
so that they don't need to be handled by the queue individually.
[Pack] Takes elements from the incoming queue, packs them in a List and puts the List in an outgoing queue. 
[Unpack] Iterates the elements from the incoming queue and puts the elements in the outgoing queue individually.
[Repack] Iterates the elements from the incoming queue, collects them in lists of a given size and puts the list in the outgoing queue.

##
Example IRL

```python
"""
Here steps for reading the h5 files and processing the calorimeter \
images to graphs are definded. `process_seq` is the function that \
should be passed the qfseq.
"""
import os
from pathlib import Path

import h5py as h5
import numpy as np
import torch_geometric
import yaml
from torch.multiprocessing import Queue

from fgsim.config import conf
from fgsim.geo.batch_stack import split_layer_subgraphs
from fgsim.geo.transform import transform

import queueflow as qf

# Load files
ds_path = Path(conf.path.dataset)
assert ds_path.is_dir()
files = [str(e) for e in sorted(ds_path.glob("**/*.h5"))]
if len(files) < 1:
    raise RuntimeError("No hdf5 datasets found")


# load lengths
if not os.path.isfile(conf.path.ds_lenghts):
    len_dict = {}
    for fn in files:
        with h5.File(fn) as h5_file:
            len_dict[fn] = len(h5_file[conf.yvar])
    with open(conf.path.ds_lenghts, "w") as f:
        yaml.dump(len_dict, f, Dumper=yaml.SafeDumper)
else:
    with open(conf.path.ds_lenghts, "r") as f:
        len_dict = yaml.load(f, Loader=yaml.SafeLoader)


# reading from the filesystem
def read_chunk(chunks):
    data_dict = {k: [] for k in conf.loader.keylist}
    for chunk in chunks:
        file_path, start, end = chunk
        with h5.File(file_path) as h5_file:
            for k in conf.loader.keylist:
                data_dict[k].append(h5_file[k][start:end])
    for k in conf.loader.keylist:
        if len(data_dict[k][0].shape) == 1:
            data_dict[k] = np.hstack(data_dict[k])
        else:
            data_dict[k] = np.vstack(data_dict[k])

    # split up the events and pass them as a dict
    output = [
        {k: data_dict[k][ientry] for k in conf.loader.keylist}
        for ientry in range(conf.loader.chunksize)
    ]
    return output


def geo_batch(list_of_graphs):
    batch = torch_geometric.data.Batch().from_data_list(list_of_graphs)
    return batch


def magic_do_nothing(elem):
    return elem


# Collect the steps
def process_seq():
    return (
        qf.ProcessStep(read_chunk, 2, name="read_chunk"),
        Queue(1),
        # In the input is now [(x,y), ... (x [300 * 51 * 51 * 25], y [300,1] ), (x,y)]
        # For these elements to be processed by each of the workers in the following
        # transformthey need to be (x [51 * 51 * 25], y [1] ):
        qf.PoolStep(
            transform, nworkers=conf.loader.num_workers_transform, name="transform"
        ),
        Queue(1),
        qf.RepackStep(conf.loader.batch_size),
        qf.ProcessStep(geo_batch, 1, name="geo_batch"),
        qf.ProcessStep(
            split_layer_subgraphs,
            conf.loader.num_workers_stack,
            name="split_layer_subgraphs",
        ),
        # Needed for outputs to stay in order.
        qf.ProcessStep(
            magic_do_nothing,
            1,
            name="magic_do_nothing",
        ),
        Queue(conf.loader.prefetch_batches),
    )
```