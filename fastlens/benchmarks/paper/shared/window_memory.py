"""Simultaneous process/allocator samples for the requested unprofiled resource window."""
import json
import os
from pathlib import Path
import threading
import time

import psutil
import torch


class WindowMemory:
    def __init__(self,output,native_stats=None):
        self.output=output
        self.native_stats=native_stats
        self.rows=[]
        self.stop=threading.Event()
        self.process=psutil.Process()

    def io_snapshot(self):
        rows=[]
        for process in [self.process,*self.process.children(recursive=True)]:
            try:
                io=process.io_counters()
                rows.append(dict(pid=process.pid,created=process.create_time(),
                                 read_bytes=io.read_bytes,read_chars=io.read_chars))
            except psutil.NoSuchProcess:
                pass
        return rows

    def sample(self):
        while not self.stop.is_set():
            rss=self.process.memory_info().rss
            tree=rss
            for child in self.process.children(recursive=True):
                try:
                    tree+=child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            pinned=torch.cuda.memory.host_memory_stats()
            row=dict(timestamp_ns=time.time_ns(),host_rss_bytes=rss,tree_rss_bytes=tree,
                torch_allocated_bytes=torch.cuda.memory_allocated(),torch_reserved_bytes=torch.cuda.memory_reserved(),
                pinned_allocated_bytes=pinned.get('allocated_bytes.current'),pinned_active_bytes=pinned.get('active_bytes.current'))
            if self.native_stats:
                row['native_pools']=self.native_stats()
            self.rows.append(row)
            self.stop.wait(.1)

    def begin(self):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.memory.reset_peak_host_memory_stats()
        self.io_before=self.io_snapshot()
        self.start_ns=time.time_ns()
        self.thread=threading.Thread(target=self.sample)
        self.thread.start()

    def end(self):
        self.end_ns=time.time_ns()
        io_after=self.io_snapshot()
        self.stop.set()
        self.thread.join()
        if not self.rows:
            raise RuntimeError('Memory window contains no sampled observations')
        record=dict(pid=os.getpid(),start_unix_ns=self.start_ns,end_unix_ns=self.end_ns,measurements=self.rows,
                    torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    io_before=self.io_before,io_after=io_after,
                    process_tree_storage_read_bytes=sum(r['read_bytes'] for r in io_after)-sum(r['read_bytes'] for r in self.io_before),
                    process_tree_read_chars=sum(r['read_chars'] for r in io_after)-sum(r['read_chars'] for r in self.io_before),
                    io_scope='Linux process-tree IO counter deltas; read_bytes is storage IO, read_chars includes non-file descriptors and excludes mmap accesses; input prefetch can cross window boundaries',
                    pinned_stats=torch.cuda.memory.host_memory_stats(),
                    scope='100ms contemporaneous snapshots; tree RSS can double-count shared pages; never sum independent high-water marks')
        (self.output/'memory_samples.json').write_text(json.dumps(record,indent=2)+'\n')
