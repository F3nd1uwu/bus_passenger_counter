"""Small CPU/RAM admission controller and process locks; no ML imports."""
import math
import os
from pathlib import Path


class ProcessLock:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open('a+b')
        try:
            if path.stat().st_size == 0:
                self.stream.write(b'0')
                self.stream.flush()
            self.stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise RuntimeError('Process lock already held')

    def close(self):
        self.stream.close()


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ''


def cpu_capacity():
    capacity = float(len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1)
    quota = read('/sys/fs/cgroup/cpu.max').split()
    if len(quota) == 2 and quota[0] != 'max':
        capacity = min(capacity, int(quota[0]) / int(quota[1]))
    return max(.1, capacity)


def memory_available_mb():
    if os.name == 'nt':
        import ctypes
        class Memory(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in ('total', 'available', 'page', 'page_available', 'virtual', 'virtual_available', 'extended')]
        value = Memory()
        value.length = ctypes.sizeof(value)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
            return 0
        return value.available / 1024 ** 2
    info = dict(line.split(':', 1) for line in read('/proc/meminfo').splitlines() if ':' in line)
    available = int(info.get('MemAvailable', '0 kB').split()[0]) * 1024
    limit, current = read('/sys/fs/cgroup/memory.max'), read('/sys/fs/cgroup/memory.current')
    if limit.isdigit() and current.isdigit():
        stat = dict(line.split() for line in read('/sys/fs/cgroup/memory.stat').splitlines())
        reclaimable = int(stat.get('inactive_file', 0))
        available = min(available, max(0, int(limit) - int(current) + reclaimable))
    return available / 1024 ** 2


def rss_mb(pid):
    for line in read(f'/proc/{pid}/status').splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) / 1024
    return 0.


def cpu_times():
    allowed = set(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else set()
    total = idle = 0
    for line in read('/proc/stat').splitlines():
        fields = line.split()
        if fields and fields[0].startswith('cpu') and fields[0][3:].isdigit():
            if allowed and int(fields[0][3:]) not in allowed:
                continue
            values = list(map(int, fields[1:9]))
            total += sum(values)
            idle += values[3] + values[4]
    return (total, idle) if total else None


def idle_cores(before, after, cores):
    if not before or not after or after[0] <= before[0]:
        return None
    return cores * max(0., min(1., (after[1]-before[1]) / (after[0]-before[0])))


def can_start(active_rss, *, max_workers, threads, memory_mb, available_mb, cpu, reserve_mb=256):
    slots = min(max_workers, math.floor(cpu / threads))
    # Reserve memory that starting processes have not allocated yet.
    pending_reservation = sum(max(0, memory_mb - rss) for rss in active_rss)
    return len(active_rss) < slots and available_mb - pending_reservation >= memory_mb + reserve_mb
