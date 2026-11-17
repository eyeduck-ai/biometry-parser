"""Sample local process resources without inventing remote or historical usage."""
from __future__ import annotations

import threading


class ResourceSampler:
    def __init__(self, pid: int, *, children: bool = True, interval: float = .05):
        self.pid, self.children, self.interval = pid, children, interval
        self.peak_rss_bytes = None
        self.cpu_seconds = None
        self._cpu = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._scope = None
        self._thread = None

    def sample(self):
        try:
            import psutil
        except ImportError:
            return
        try:
            root = psutil.Process(self.pid)
            processes = [root] + (root.children(recursive=True) if self.children else [])
            rss, cpu, count = 0, {}, 0
            for process in processes:
                try:
                    rss += process.memory_info().rss
                    times = process.cpu_times()
                    cpu[(process.pid, process.create_time())] = times.user + times.system
                    count += 1
                except psutil.Error:
                    pass
            if not count:
                return
        except (psutil.Error, OSError):
            return
        with self._lock:
            for process, seconds in cpu.items():
                self._cpu[process] = max(self._cpu.get(process,0.),seconds)
            self.cpu_seconds = sum(self._cpu.values())
            self.peak_rss_bytes = max(self.peak_rss_bytes or 0, rss)
            if self._scope is not None:
                self._scope['peak_rss_bytes'] = max(self._scope.get('peak_rss_bytes') or 0, rss)
                self._scope['samples'] += 1

    def start(self):
        self.sample()
        def monitor():
            while not self._stop.wait(self.interval):
                self.sample()
        self._thread = threading.Thread(target=monitor, daemon=True)
        self._thread.start()

    def begin_scope(self):
        self.sample()
        with self._lock:
            self._scope = {'peak_rss_bytes': None, 'samples': 0, 'cpu_start': self.cpu_seconds}
        self.sample()

    def end_scope(self):
        self.sample()
        with self._lock:
            scope, self._scope = self._scope, None
            if scope is None:
                return {'peak_rss_bytes': None, 'cpu_seconds': None, 'resource_samples': 0}
            cpu = None if self.cpu_seconds is None or scope['cpu_start'] is None else max(0., self.cpu_seconds-scope['cpu_start'])
            return {'peak_rss_bytes': scope['peak_rss_bytes'], 'cpu_seconds': cpu,
                    'resource_samples': scope['samples'], 'sampling_interval_seconds': self.interval}

    def close(self):
        self.sample()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
