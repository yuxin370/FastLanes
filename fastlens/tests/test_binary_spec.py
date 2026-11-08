"""Read small format vectors independently using the normative Markdown field tables."""
import json
from pathlib import Path
import re
import struct
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / 'fastlens/docs/spec'
VECTORS = Path(__file__).parent / 'fixtures/formats'


def crc64(data, zero_field=None):
    # Bitwise reference, independent of the production table-based implementation.
    crc = 0
    for index, byte in enumerate(data):
        if zero_field is not None and zero_field <= index < zero_field + 8:
            byte = 0
        crc ^= byte << 56
        for _ in range(8):
            crc = ((crc << 1) ^ (0x42f0e1eba9ea3693 if crc >> 63 else 0)) & ((1 << 64) - 1)
    return crc


def header(data, document, table=0):
    """Use the published offsets and types, so a wrong table fails on real bytes."""
    source = (SPEC / document).read_text()
    tables = re.findall(r'\| Offset \| Type \| Meaning \|\n[^\n]+\n((?:\|[^\n]+\n)+)', source)
    values = {}
    for line in tables[table].splitlines():
        position, kind, meaning = [cell.strip() for cell in line.split('|')[1:4]]
        offset = int(position.split('..')[0])
        if kind == '6 × (u64 offset, u64 size)':
            kind = 'u64[12]'
        match = re.fullmatch(r'(byte|[ui](?:8|16|32|64))(?:\[(\d+)\])?', kind)
        if match is None:
            raise ValueError(f'unsupported field type in {document}: {kind}')
        base, count = match[1], int(match[2] or 1)
        if base == 'byte':
            value = data[offset:offset + count]
            assert len(value) == count
            if offset == 0:
                magic = re.search(r'`([^`]+)`', meaning)[1].replace('\\0', '\0').encode('ascii')
                assert value == magic, (document, value, magic)
            value = value.hex()
        else:
            codes = {'u8': 'B', 'u16': 'H', 'u32': 'I', 'u64': 'Q',
                     'i8': 'b', 'i16': 'h', 'i32': 'i', 'i64': 'q'}
            decoded = struct.unpack_from('<' + codes[base] * count, data, offset)
            value = decoded[0] if count == 1 else list(decoded)
            constant = re.search(r'(?:format version, (?:currently )?|metadata version, |version, |header bytes, )`?(\d+)', meaning)
            if constant:
                assert value == int(constant[1]), (document, offset, value, constant[1])
        values[str(offset)] = value
    return values


class Cursor:
    def __init__(self, data, offset):
        self.data, self.offset = data, offset

    def take(self, fmt):
        result = struct.unpack_from('<' + fmt, self.data, self.offset)
        self.offset += struct.calcsize('<' + fmt)
        return result

    def string(self):
        size, = self.take('I')
        value = self.data[self.offset:self.offset + size]
        assert len(value) == size
        self.offset += size
        return value.decode('utf-8')


def describe_store(root):
    manifest_bytes = (root / 'manifest.bin').read_bytes()
    manifest = header(manifest_bytes, 'MANIFEST.md')
    cursor = Cursor(manifest_bytes, 34)
    partitions = []
    for _ in range(manifest['30']):
        fields = cursor.take('IQIQQQIIQQ')
        fls, metadata = cursor.string(), cursor.string()
        assert fields[8] == (root / fls).stat().st_size
        data = (root / metadata).read_bytes()
        assert fields[9] == len(data)
        meta = header(data, 'IMAGE_METADATA.md')
        assert meta['18'] == fields[2]
        sections, section_cursor = [], Cursor(data, 38)
        while section_cursor.offset < len(data):
            kind, size = section_cursor.take('HQ')
            start = section_cursor.offset
            assert start + size <= len(data)
            widths = {1: 28, 5: 152, 6: 36, 7: 24, 2: 17 + 25 * meta['26']}
            if kind in widths:
                assert size % widths[kind] == 0
                if kind in (1, 5, 2):
                    assert size // widths[kind] == meta[{1: '26', 5: '30', 2: '18'}[kind]]
            sections.append([kind, size])
            section_cursor.offset += size
        partitions.append(dict(id=fields[0], first_image=fields[1], images=fields[2],
                               real_rows=fields[3], rowgroups=fields[6], fls=fls,
                               metadata_header=meta, metadata_sections=sections))
    result = dict(manifest_header=manifest, partitions=partitions)
    if manifest['8'] == 1:
        assert cursor.offset == len(manifest_bytes)
        index_bytes = (root / 'geometry/index.bin').read_bytes()
        index = header(index_bytes, 'GEOMETRY.md', 1)
        assert len(index_bytes) == index['40'] == 64 + 32 * len(partitions)
        assert index['16'] == crc64(manifest_bytes)
        assert index['24'] == len(manifest_bytes)
        assert index['48'] == crc64(index_bytes, 48)
        result['index_header'] = index
        for order, partition in enumerate(partitions):
            fields = header(index_bytes[64 + 32 * order:64 + 32 * (order + 1)], 'GEOMETRY.md', 2)
            record = tuple(fields[str(offset)] for offset in (0, 4, 8, 16, 24))
            data = (root / f"geometry/{partition['id']:06d}.bin").read_bytes()
            geometry = header(data, 'GEOMETRY.md')
            assert record[:3] == (partition['id'], partition['images'], partition['first_image'])
            assert record[3:] == (len(data), geometry['24'])
            assert geometry['16'] == len(data)
            assert geometry['24'] == crc64(data, 24)
            assert geometry['32'] == index['16']
            assert geometry['92'] == partition['images']
            offsets = geometry['144'] + [len(data)]
            assert offsets == sorted(offsets) and offsets[0] == 256
            for position, count_key, width in ((0, '108', 64), (2, '128', 8), (4, '132', 24),
                                               (5, '92', 24), (6, '112', 24), (7, '116', 136)):
                assert offsets[position + 1] - offsets[position] == geometry[count_key] * width
            partition['geometry_header'] = geometry
    else:
        assert manifest['8'] == 3
        assert manifest_bytes[cursor.offset:cursor.offset + 8] == b'GJDCCV31'
        cursor.offset += 8
        assert cursor.string() == 'image-major-vector-rowgroups'
        assert cursor.string() == 'galp-compact-v1'
        assert cursor.take('I') == (1024,)
        assert cursor.string() == 'tiled-z32'
        assert cursor.take('HI') == (3, len(partitions))
        for partition in partitions:
            part_id, payload_size, payload_crc, descriptor_size, _ = cursor.take('IQQQQ')
            assert part_id == partition['id']
            fls = (root / partition['fls']).read_bytes()
            footer_offset, footer_size = struct.unpack_from('<QQ', fls, len(fls) - 24)
            assert footer_offset == 24 + payload_size and footer_size == descriptor_size
            data = fls[footer_offset:footer_offset + descriptor_size]
            compact = header(data, 'IMAGE_MAJOR.md')
            assert compact['16'] == len(data) and compact['24'] == payload_size
            assert compact['168'] == crc64(data, 168)
            assert compact['32'] == payload_crc == crc64(fls[24:footer_offset])
            sections = list(zip(compact['72'][::2], compact['72'][1::2]))
            assert all(start % 8 == 0 and start >= 256 and start + size <= len(data)
                       for start, size in sections)
            assert sections[1][1] == compact['64'] * 32
            assert sections[2][1] == compact['68'] * 32
            assert sections[3][1] == compact['40'] * 48
            assert sections[4][1] == compact['40'] * compact['48'] * (4 if compact['12'] & 32 else 8)
            partition['compact_header'] = compact
        assert cursor.offset == len(manifest_bytes)
    # Native generation directory names are incidental; fields are not.
    for partition in partitions:
        del partition['fls']
    return result


class BinarySpecificationTests(unittest.TestCase):
    def test_published_vectors_match_normative_fields(self):
        for layout in ('block-major', 'image-major'):
            with self.subTest(layout=layout):
                root = VECTORS / layout
                expected = json.loads((root / 'expected.json').read_text())
                self.assertEqual(describe_store(root), expected)


if __name__ == '__main__':
    unittest.main()
