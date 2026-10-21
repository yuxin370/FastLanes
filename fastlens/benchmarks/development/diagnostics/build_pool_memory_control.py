"""Build a postprocessing-only allocation control without relinking the native reader."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[4]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    original = ROOT / 'fastlens/src/api/direct_dct_pls_postprocess.cu'
    text = original.read_text()
    text = ('#include <cstdlib>\n#include <atomic>\n'
            'namespace { std::atomic<unsigned long long> allocation_control_calls {0}; }\n'
            'extern "C" unsigned long long pool_allocation_control_calls() { return allocation_control_calls.load(); }\n') + text
    marker = '\tstd::optional<GPUArray<float>>           targets_output;'
    assert text.count(marker) == 1
    text = text.replace(marker, marker + '\n\tstd::optional<GPUArray<float>> control_y, control_c;')
    marker = '\t\tconst auto       capacity    = std::min<size_t>(microbatch_images, labels.size());'
    assert text.count(marker) == 1
    text = text.replace(marker, marker + '\n\t\tconst bool pool_control = std::getenv("FASTLENS_POOL_WORKSPACE_CONTROL") != nullptr;'
        '\n\t\tconst auto allocation_images = pool_control ? labels.size() : capacity;'
        '\n\t\t++allocation_control_calls;')
    for component in ('y', 'c'):
        for buffer in ('a', 'b'):
            marker = f'impl_->{component}_{buffer}.emplace(capacity * {component}_per_image, stream);'
            assert text.count(marker) == 1
            text = text.replace(marker, marker.replace('capacity *', 'allocation_images *'))
    marker = '\t\tconst auto [y_output, c_output] = source.device_batch().transform_workspace_async();'
    assert text.count(marker) == 1
    text = text.replace(marker,
        '\t\tauto [y_output, c_output] = source.device_batch().transform_workspace_async();\n'
        '\t\tif (pool_control) {\n'
        '\t\t\timpl_->control_y.emplace(labels.size() * y_per_image, stream);\n'
        '\t\t\timpl_->control_c.emplace(labels.size() * c_per_image, stream);\n'
        '\t\t\ty_output = impl_->control_y->get();\n'
        '\t\t\tc_output = impl_->control_c->get();\n'
        '\t\t}')
    source = args.output.resolve() / 'pool_memory_control.cu'
    source.write_text(text)
    commands = json.loads((ROOT / 'build/compile_commands.json').read_text())
    spec, = [c for c in commands if c['file'] == str(original)]
    command = shlex.split(spec['command'])
    command = [str(source) if item == str(original) else item for item in command if item != '-rdc=true']
    command[command.index('-c')] = '-shared'
    command[command.index('-o') + 1] = str(args.output.resolve() / 'libpool_memory_control.so')
    command += ['--cudart=shared', '-Xlinker=-rpath,' + str(Path(command[0]).resolve().parent.parent / 'lib')]
    subprocess.run(command, cwd=spec['directory'], check=True)


if __name__ == '__main__':
    main()
