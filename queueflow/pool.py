from collections.abc import Iterable
from multiprocessing.queues import Empty

from torch import multiprocessing as mp

from .logger import logger

from .step_base import StepBase
from .terminate_queue import TerminateQueue


class PoolStep(StepBase):
    """Class for simple processing steps pooled over multiple workes.
    Each incoming object is processed by a multiple subprocesses
    per worker into a single outgoing element."""

    def __init__(
        self,
        *args,
        nworkers: int,
        **kwargs,
    ):
        # Spawn only one process with deamonize false that can spawn the Pool
        kwargs["deamonize"] = False

        # Make sure the contructor of the base class only initializes
        # one process that manages the pool
        self.n_pool_workers = nworkers
        kwargs["nworkers"] = 1
        super().__init__(*args, **kwargs)

    def start(self):
        for p in self.processes:
            p.daemon = self.deamonize
            p.start()

    def stop(self):
        for p in self.processes:
            if p.is_alive():
                p.join(5)
                p.kill()

    def process_status(self):
        return (
            sum([p.is_alive() for p in self.processes]) * self.n_pool_workers,
            self.n_pool_workers,
        )

    def _worker(self):
        self.set_workername()
        logger.debug(
            f"{self.workername} pool  initalizing with"
            f" {self.n_pool_workers} subprocesses"
        )
        self.pool = mp.Pool(self.n_pool_workers)

        while not self.shutdown_event.is_set():
            try:
                wkin = self.inq.get(block=True, timeout=0.05)
            except Empty:
                continue
            logger.debug(
                f"""\
{self.workername} working on element of type {type(wkin)} from queue {id(self.inq)}."""
            )

            # If the process gets a TerminateQueue object,
            # it terminates the pool and and puts the terminal element in
            # in the outgoing queue.
            if isinstance(wkin, TerminateQueue):
                logger.info(f"{self.workername} terminating")
                self.safe_put(self.outq, TerminateQueue())
                logger.warning(
                    f"""\
{self.workername} finished with iterable (in {self.count_in}/out {self.count_out})"""
                )
                self.count_in, self.count_out = 0, 0
                continue
            self.count_in += 1
            wkin = self._clone_tensors(wkin)

            assert isinstance(wkin, Iterable)
            logger.debug(
                f"{self.workername} got element"
                + f" {id(wkin)} of element type {type(wkin)}."
            )

            try:
                wkout_async_res = self.pool.map_async(
                    self.workerfn,
                    wkin,
                )
                while True:
                    if wkout_async_res.ready():
                        wkout = wkout_async_res.get()
                        break
                    elif self.shutdown_event.is_set():
                        break
                    wkout_async_res.wait(1)
                if self.shutdown_event.is_set():
                    break

            except Exception as error:
                logger.warning(f"""{self.workername} got error""")
                self.handle_error(error, wkin)
                break

            logger.debug(
                f"""\
{self.workername} push pool output list {id(wkout)} with \
element type {type(wkin)} into output queue {id(self.outq)}."""
            )
            # Put while there is no shutdown event
            self.safe_put(self.outq, wkout)
            self.count_out += 1
            del wkin
        self.pool.close()
        self.pool.terminate()
        logger.debug(f"""{self.workername} pool closed""")
        self.outq.cancel_join_thread()
        self._close_queues()
        logger.debug(f"""{self.workername} queues closed""")
