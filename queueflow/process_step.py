from multiprocessing.queues import Empty

from torch import multiprocessing as mp

from .logger import logger

from .step_base import StepBase
from .terminate_queue import TerminateQueue


class ProcessStep(StepBase):
    """Class for simple processing steps.
    Each incoming object is processed by a
    single worker into a single outgoing element."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.finish_barrier = mp.Barrier(parties=self.nworkers)
        self.sync_barrier = mp.Barrier(parties=self.nworkers)

    def __handle_terminal(self):
        logger.debug(f"{self.workername}  Got terminal element.")

        # Put the terminal element back in the input queue
        self.safe_put(self.inq, TerminateQueue())

        # Make the first worker to reach the terminal element
        # aquires the lock and waits for the other processes
        # processes to finish
        logger.debug(f"{self.workername} waiting at barrier.")
        self.sync_barrier.reset()
        if self.finish_barrier.wait() == 0:
            assert isinstance(self.inq.get(), TerminateQueue)
            self.safe_put(self.outq, TerminateQueue())
            logger.debug(f"{self.workername} put terminal element in outq.")
        if self.sync_barrier.wait() == 0:
            self.finish_barrier.reset()

        logger.debug(
            f"""\
{self.workername} finished with iterable (in {self.count_in}/out {self.count_out})"""
        )
        self.count_in, self.count_out = 0, 0

    def _worker(self):
        self.set_workername()

        logger.debug(
            f"{self.workername} start reading from input queue {id(self.inq)}."
        )
        while not self.shutdown_event.is_set():
            try:
                wkin = self.inq.get(block=True, timeout=0.05)
            except Empty:
                continue
            logger.debug(
                f"""\
{self.workername} working on element of type {type(wkin)} from queue {id(self.inq)}."""
            )
            # If the process gets the terminate_queue object,
            # wait for the others and put it in the next queue
            if isinstance(wkin, TerminateQueue):
                self.__handle_terminal()
                continue
            self.count_in += 1

            # We need to overwrite the method of cloning the batches
            # because we have list of tensors as attibutes of the batch.
            # If copy.deepcopy is called on this object
            wkin = self._clone_tensors(wkin)

            try:
                wkout = self.workerfn(wkin)

            # Catch Errors in the worker function
            except Exception as error:
                self.handle_error(error, wkin)
                break

            logger.debug(
                f"{self.workername} push single "
                + f"output of type {type(wkout)} into output queue {id(self.outq)}."
            )
            self.safe_put(self.outq, wkout)
            self.count_out += 1
            del wkin
        self._close_queues()
