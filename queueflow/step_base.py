import threading
import traceback
from multiprocessing.queues import Full

import torch
import torch_geometric
from torch import multiprocessing as mp

from .batch_utils import batch_to_numpy_dict
from .handle_data import HandleDataBase
from .logger import logger


class StepBase(HandleDataBase):
    """Base class"""

    def __init__(
        self,
        shutdown_event: mp.Event,
        workerfn: callable = None,
        nworkers: int = 1,
        deamonize: bool = True,
        name: str = "DefaultWorkerName",
    ):
        self.name = type(self) if name is None else name
        self.workerfn = workerfn
        self.nworkers = nworkers
        self.deamonize = deamonize
        self.processes = [
            mp.Process(
                target=self._worker,
                daemon=self.deamonize,
                args=(shutdown_event,),
            )
            for _ in range(self.nworkers)
        ]
        self.count_in = 0
        self.count_out = 0
        self.marked_as_working = False
        self.shutdown_event = shutdown_event

    def connect_to_sequence(self, input_queue, output_queue, error_queue):
        self.inq = input_queue
        self.outq = output_queue
        self.error_queue = error_queue

    def _close_queues(self):
        self.outq.close()
        self.outq.join_thread()
        logger.debug(f"""{self.workername} outq closed""")
        self.inq.close()
        self.inq.join_thread()
        logger.debug(f"""{self.workername} inq closed""")
        self.error_queue.close()
        self.error_queue.join_thread()
        logger.debug(f"""{self.workername} error_queue closed""")

    def set_workername(self):
        self.workername = self.name + "-" + mp.current_process().name.split("-")[1]
        mp.current_process().name = self.workername
        threading.current_thread().name = "MainThread-" + self.workername

    def start(self):
        for p in self.processes:
            p.start()

    def stop(self):
        for p in self.processes:
            if p.is_alive():
                p.join(5)
                if p.exitcode is None:
                    p.kill()
                    logger.warning(
                        f"""\
Had to kill process of name {self.name}."""
                    )
                p.join(0)

    def safe_put(self, queue, element):
        # element = self._clone_tensors(element)
        while not self.shutdown_event.is_set():
            try:
                queue.put(element, True, 1)
                break
            except Full:
                continue
            except KeyboardInterrupt:
                break

    def process_status(self):
        return (sum([p.is_alive() for p in self.processes]), self.nworkers)

    def handle_error(self, error, obj):
        tb = traceback.format_exc()

        if isinstance(obj, torch.Tensor):
            obj = obj.numpy()
        if isinstance(obj, torch_geometric.data.Data):
            obj = batch_to_numpy_dict(obj)

        workermsg = f"""
{self.workername} failed on element of type of type {type(obj)}."""
        self.error_queue.put((workermsg, obj, str(error), tb))

    def _worker(self, shutdown_event):
        raise NotImplementedError
