"""Figure 11a: five DCT profiles against the six RGB baselines in the final PDF."""
from fastlens.benchmarks.paper.end_to_end import main

if __name__ == '__main__':
    main('deployment', 'section_6_2_3/deployment', 'plot_deployment.py')
