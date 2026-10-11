"""Persistent CPU workers for independent determinized UCT trees.

Only complete sampled worlds and RNG states cross the process boundary.
The framed pickle stream connects our own local child processes, never a
network endpoint or external data source.
"""

import pickle
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from random import Random
from threading import Lock

from skat.agents.search.uct import uct_search
from skat.utils.processes import available_cpus, worker_environment


def _search(tasks):
    results = []
    for world, iterations, rng_state in tasks:
        rng = Random()
        rng.setstate(rng_state)
        results.append(uct_search(world, iterations, rng))
    return results


def _send(stream, value):
    data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack("!I", len(data)))
    stream.write(data)
    stream.flush()


def _receive(stream):
    header = stream.read(4)
    if len(header) != 4:
        raise RuntimeError("MCTS worker connection closed")
    size = struct.unpack("!I", header)[0]
    data = stream.read(size)
    if len(data) != size:
        raise RuntimeError("incomplete MCTS worker result")
    return pickle.loads(data)


class SearchPool:
    """One lazily started pool shared by all MCTS seats of a table."""

    def __init__(self, workers=None, *, capacity=20):
        if workers is not None and (type(workers) is not int or workers < 1):
            raise ValueError("workers must be a positive integer")
        if type(capacity) is not int or capacity < 1:
            raise ValueError("search capacity must be a positive integer")
        self.workers = min(workers or available_cpus(), capacity)
        self.processes = []
        self.executor = None
        self.lock = Lock()
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close(force=exc_type is not None)

    def _start(self, count):
        try:
            for _ in range(count):
                self.processes.append(
                    subprocess.Popen(
                        [sys.executable, "-m", "skat.agents.search.parallel"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        env=worker_environment(),
                    )
                )
            self.executor = ThreadPoolExecutor(max_workers=count)
        except BaseException:
            self.close(force=True)
            raise

    @staticmethod
    def _exchange(process, tasks):
        try:
            _send(process.stdin, tasks)
            return _receive(process.stdout)
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError("MCTS search worker failed") from error

    def search(self, tasks):
        with self.lock:
            if self.closed:
                raise RuntimeError("MCTS search pool is closed")
            count = min(self.workers, len(tasks))
            if count <= 1:
                return _search(tasks)
            if not self.processes:
                self._start(count)
            count = min(count, len(self.processes))
            batches = [tasks[i::count] for i in range(count)]
            futures = [
                self.executor.submit(self._exchange, p, batch)
                for p, batch in zip(self.processes, batches)
            ]
            try:
                results = [None] * len(tasks)
                for i, future in enumerate(futures):
                    results[i::count] = future.result()
                return results
            except BaseException:
                self.close(force=True)
                raise

    def close(self, force=False):
        if self.closed:
            return
        self.closed = True
        for process in self.processes:
            if process.poll() is None:
                try:
                    if force:
                        process.terminate()
                    else:
                        _send(process.stdin, None)
                except (BrokenPipeError, OSError):
                    process.terminate()
        deadline = time.monotonic() + 5
        for process in self.processes:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            # Buffered writes can fail again on close after a child crashes.
            # Retain the search error and still reap every other child.
            with suppress(OSError):
                process.stdin.close()
            process.stdout.close()
        if self.executor is not None:
            self.executor.shutdown(wait=True)


if __name__ == "__main__":
    while True:
        tasks = _receive(sys.stdin.buffer)
        if tasks is None:
            break
        _send(sys.stdout.buffer, _search(tasks))
