#!/usr/bin/env python3
"""Verify FFCV ordinals/labels exhaustively and decoded pixels on three samples."""
import json
import argparse
from pathlib import Path
import time

import numpy as np
from PIL import Image
from ffcv.reader import Reader
from ffcv.loader import Loader,OrderOption
from ffcv.fields.decoders import SimpleRGBImageDecoder,IntDecoder

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'results/preparation/p0_ffcv'


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    OUT=parser.parse_args().output
    if (OUT/'verification.json').exists():
        raise FileExistsError('Reuse the existing verification instead of overwriting it')
    conversion=json.loads((OUT/'conversion.json').read_text())
    entries=json.loads((OUT/'samples.json').read_text())
    started=time.perf_counter()
    reader=Reader(str(OUT/'train_window.beton'))
    assert reader.field_names==['image','label','ordinal']
    assert reader.num_samples==len(entries)==conversion['samples']
    assert np.array_equal(reader.metadata['f1'],np.asarray([s['label'] for s in entries]))
    assert np.array_equal(reader.metadata['f2'],np.arange(len(entries)))
    chosen=np.asarray([0,len(entries)//2,len(entries)-1])
    loader=Loader(str(OUT/'train_window.beton'),batch_size=1,num_workers=1,order=OrderOption.SEQUENTIAL,
        drop_last=False,indices=chosen,pipelines={'image':[SimpleRGBImageDecoder()],
        'label':[IntDecoder()],'ordinal':[IntDecoder()]})
    checked=[]
    for images,labels,ordinals in loader:
        i=int(ordinals[0,0])
        with Image.open(entries[i]['path']) as source:
            assert np.array_equal(images[0],np.asarray(source.convert('RGB')))
        assert int(labels[0,0])==entries[i]['label']
        checked.append(i)
    assert checked==chosen.tolist()
    seconds=time.perf_counter()-started
    stat=(OUT/'train_window.beton').stat()
    external=(OUT/'samples.json').stat()
    payload=int(reader.alloc_table['size'].sum())
    metadata=reader.header.nbytes+reader.field_descriptors.nbytes+reader.metadata.nbytes
    indexes=reader.alloc_table.nbytes
    padding=stat.st_size-payload-metadata-indexes
    assert padding>=0
    result=dict(passed=True,samples=int(reader.num_samples),labels_and_ordinals_verified=int(reader.num_samples),
        pixel_sample_ordinals=checked,pixel_sample_ids=[entries[i]['logical_sample_id'] for i in checked],
        verification_s=seconds,scope='All labels/ordinals, three exact decoded-RGB pixel comparisons; not exhaustive image-content verification',
        payload_bytes=payload,embedded_metadata_bytes=metadata,index_bytes=indexes,padding_bytes=padding,
        external_metadata_bytes=external.st_size,beton_bytes=stat.st_size,
        representation_bytes=stat.st_size+external.st_size,allocated_bytes=(stat.st_blocks+external.st_blocks)*512)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
