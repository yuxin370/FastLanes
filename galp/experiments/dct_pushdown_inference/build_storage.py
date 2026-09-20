"""Build the experiment bridge using the configured project's compiler and libraries."""
from pathlib import Path
import shlex
import subprocess
root = Path(__file__).resolve().parents[3]
source = Path(__file__).with_name('dctnet_storage.cpp')
target = source.with_suffix('')
config = root / 'build/galp/tools/jpeg_dct/CMakeFiles/galp_jpeg_dct_tool.dir'
options = []
for line in (config / 'flags.make').read_text().splitlines():
    if line.startswith(('CXX_DEFINES =', 'CXX_INCLUDES =', 'CXX_FLAGS =')):
        options += shlex.split(line.split('=', 1)[1])
subprocess.run(['/usr/bin/clang++', *options, '-c', str(source), '-o', str(source.with_suffix('.o'))], check=True)
link = shlex.split((config / 'link.txt').read_text())
link = [str(source.with_suffix('.o')) if arg.endswith('jpeg_dct_tool.cpp.o') else str(target) if arg == 'galp_jpeg_dct_tool' else arg for arg in link]
subprocess.run(link, cwd=config.parent.parent, check=True)
