#decompress table
/ path / to / FastLanes / build / galp / tools / galp_cli read_table / path / to / FastLanes / data / fls / galp -
    test / data.fls / tmp / out.csv

#only decompress one specific                                     rowgroup
        / path / to / FastLanes / build / galp / tools / galp_cli read_table / path / to / FastLanes / data / fls /
        galp -
    test / data.fls / tmp / out.csv-- rowgroup 0

#benchmark：only GPU decompression, no materialize(mega kernel)
        / path / to / FastLanes / build / galp / tools / galp_cli benchmark / path / to / FastLanes / data / fls /
        galp -
    test / data.fls-- samples 100

#benchmark：full table, non mega kernel(one kernel per rowgroup& data - type)
        / path / to / FastLanes / build / galp / tools / galp_cli benchmark / path / to / FastLanes / data / fls /
        galp -
    test / data.fls-- samples 100 --no - mega -
    kernel

#benchmark：only GPU decompression, with kernel launch overhead(mega kernel)
        / path / to / FastLanes / build / galp / tools / galp_cli benchmark / path / to / FastLanes / data / fls /
        galp -
    test / data.fls-- samples 100 --estimate - launch-- launch -
    iters 10000

#benchmark：only GPU decompression, with kernel launch overhead(only one rowgroup, multi kernel)
        / path / to / FastLanes / build / galp / tools / galp_cli benchmark / path / to / FastLanes / data / fls /
        galp -
    test / data.fls-- samples 100 --estimate - launch-- launch - iters 10000 --rowgroup 0
