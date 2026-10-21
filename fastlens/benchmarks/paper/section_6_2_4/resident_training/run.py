"""Figure 11b: restored training windows versus eight real GPU-resident batches."""
from fastlens.benchmarks.paper.end_to_end import main

if __name__ == '__main__':
    main('resident', 'section_6_2_4/resident_training',
         'plot_training_resident_attainment.py')
