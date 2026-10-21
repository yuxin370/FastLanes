"""Execute a warm-cache pass and retain concurrent-load observations."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from fastlens.benchmarks.execution.resource_guard import ResourceGuard
from fastlens.benchmarks.execution.resource_guard import owned_pids
from fastlens.benchmarks.profiling.input_motivation import cache_pages


def run(command, output, env, paths, *, bind_to_gpu=False):
    guard=ResourceGuard(env['CUDA_VISIBLE_DEVICES'])
    if bind_to_gpu:
        prefix=['numactl', f'--cpunodebind={guard.numa_node}', f'--membind={guard.numa_node}']
        command=[*prefix, *command]
    while guard.blockers(owned_pids()):
        print('Target GPU occupied:',guard.observations,flush=True)
        time.sleep(5)
    if paths is not None:
        if bind_to_gpu:
            helper = ('import json,sys; from pathlib import Path; '
                      'from fastlens.benchmarks.profiling.input_motivation import cache_pages; '
                      'print(json.dumps(cache_pages([Path(p) for p in json.load(sys.stdin)],warm=True)))')
            cache=json.loads(subprocess.check_output([*prefix, sys.executable, '-c', helper],
                input=json.dumps([str(p) for p in paths]), text=True, env=env))
            cache['warmup_numa_node']=guard.numa_node
        else:
            cache=cache_pages(paths,warm=True)
        (output/'warm_cache.json').write_text(json.dumps(cache,indent=2)+'\n')
        if cache['resident_fraction']!=1:
            raise RuntimeError('Input pages not fully resident')
    print('START',output,flush=True)
    concurrent=False
    with (output/'concurrent_load.jsonl').open('x') as stream:
        process=subprocess.Popen([sys.executable,'fastlens/benchmarks/execution/run_logged.py','--log',str(output/'run.log'),'--',*command],
            env=env,start_new_session=True)
        try:
            while process.poll() is None:
                blocked=guard.blockers(owned_pids())
                concurrent=concurrent or any('pid' in o for o in guard.observations)
                stream.write(json.dumps(dict(timestamp=time.time(),observations=guard.observations,
                    load_average=os.getloadavg(),pressure={k:Path('/proc/pressure',k).read_text() for k in ('cpu','memory','io')}))+'\n')
                stream.flush()
                if blocked:
                    raise RuntimeError(f'Target GPU contention: {blocked}; own run stopped')
                time.sleep(5)
            stream.write(json.dumps(dict(timestamp=time.time(),phase='process_finished',
                returncode=process.returncode))+'\n')
            stream.flush()
        finally:
            # A failed monitor must not leave its detached measurement running.
            if process.poll() is None:
                try:
                    os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass  # The owned process group exited between poll and kill.
                process.wait()
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode,command)
    return concurrent
