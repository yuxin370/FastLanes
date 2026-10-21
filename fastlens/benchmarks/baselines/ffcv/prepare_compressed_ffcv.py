"""Pack existing JPEG bytes into standard FFCV image fields without re-encoding."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image


def encode_original_jpeg(destination, path, malloc):
    path = Path(path)
    with Image.open(path) as image:
        if image.format != 'JPEG' or image.mode != 'RGB':
            raise ValueError(f'Expected RGB JPEG: {path}')
        width, height = image.size
    encoded = np.fromfile(path, dtype=np.uint8)
    destination['mode'] = 0  # FFCV IMAGE_MODES['jpg']
    destination['height'], destination['width'] = height, width
    destination['data_ptr'], storage = malloc(encoded.size)
    storage[:] = encoded


def original_jpeg_field():
    from ffcv.fields import RGBImageField
    field = RGBImageField(write_mode='jpg')
    # Keep the standard field type/metadata so ordinary FFCV loaders can read it.
    field.encode = encode_original_jpeg
    return field


class EncodedImages:
    def __init__(self, entries, training):
        self.entries, self.training = entries, training

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        if self.training:
            return entry['path'], int(entry['label']), index
        return entry['path'], int(entry['ordinal'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    from ffcv.fields import IntField
    from ffcv.writer import DatasetWriter
    from ffcv.reader import Reader
    entries = json.loads(args.samples.read_text())
    if isinstance(entries, dict):
        entries = entries['samples']
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / ('train_window.beton' if args.training else 'val.beton')
    if path.exists():
        raise FileExistsError(path)
    fields = {'image': original_jpeg_field()}
    if args.training:
        fields['label'] = IntField()
    fields['ordinal'] = IntField()
    started = time.perf_counter()
    DatasetWriter(str(path), fields, num_workers=args.workers).from_indexed_dataset(
        EncodedImages(entries, args.training))
    seconds = time.perf_counter() - started
    (args.output / 'samples.json').write_text(json.dumps(entries) + '\n')
    reader = Reader(str(path))
    assert reader.num_samples == len(entries)
    assert np.all(reader.metadata['f0']['mode'] == 0)
    ordinal_field = 'f2' if args.training else 'f1'
    expected = np.arange(len(entries)) if args.training else [s['ordinal'] for s in entries]
    np.testing.assert_array_equal(reader.metadata[ordinal_field], expected)
    if args.training:
        np.testing.assert_array_equal(reader.metadata['f1'], [s['label'] for s in entries])
    # Verify the complete stored JPEG payload, outside every performance timer.
    with path.open('rb') as stream:
        for entry, field in zip(entries, reader.metadata['f0']):
            source = Path(entry['path']).read_bytes()
            stream.seek(int(field['data_ptr']))
            assert stream.read(len(source)) == source
    record = dict(format='FFCV JPEG; original ImageNet-512 JPEG bytes, no re-encoding', write_mode='jpg',
                  source=str(args.samples), samples=len(entries), conversion_s=seconds,
                  jpeg_bytes=sum(Path(e['path']).stat().st_size for e in entries),
                  representation_bytes=path.stat().st_size,
                  metadata_bytes=(args.output / 'samples.json').stat().st_size,
                  payload_verification='All stored JPEG bytes, labels and ordinals verified',
                  workers=args.workers)
    (args.output / 'conversion.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
