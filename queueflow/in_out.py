from multiprocessing.queues import Empty, Full

import torch_geometric

from .handle_data import HandleDataBase
from .logger import logger
from .terminate_queue import TerminateQueue


class InOutStep(HandleDataBase):
    def __init__(self):
        raise NotImplementedError

    def safe_put(self, queue, element):
        while not self.shutdown_event.is_set():
            try:
                queue.put(element, True, 1)
                break
            except Full:
                continue


class InputStep(InOutStep):
    """Internal class to read in the iterable into a the first queue"""

    def __init__(self, shutdown_event):
        self.name = "input step"
        self.shutdown_event = shutdown_event

    def queue_iterable(self, iterable_object):
        assert hasattr(iterable_object, "__iter__")
        i = 0
        for element in iterable_object:
            self.safe_put(self.outq, element)
            i = i + 1
        logger.debug(f"Queuing {i} elements complete")
        self.safe_put(self.outq, TerminateQueue())

    def connect_to_sequence(self, output_queue):
        self.outq = output_queue


class OutputStep(InOutStep):
    """Internal generator class to returning the outputs from the last queue."""

    def __init__(self, shutdown_event):
        self.name = "output step"
        self.shutdown_event = shutdown_event

    def start(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        while not self.shutdown_event.is_set():
            try:
                out = self.inq.get(block=True, timeout=0.05)
                if isinstance(out, TerminateQueue):
                    logger.debug("OutputStep got terminal element.")
                    break
                return self._clone_tensors(out)
            except Empty:
                continue
            logger.debug("Sequence output ready.")
        raise StopIteration

    def connect_to_sequence(self, input_queue):
        self.inq = input_queue
