"""Figure 9: seven matched DCT models, two GPUs, training and inference."""
from fastlens.benchmarks.paper.end_to_end import main

if __name__ == '__main__':
    main('matched', 'section_6_2_1/matched_dct', 'plot_matched_dct.py')
