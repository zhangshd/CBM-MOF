"""Process-validation helpers for isotherm fitting and PSA/VSA simulation.

Entry point:
    select_optimization_candidates - Single command: select candidates + build Adsorbents CSVs
                                     + generate ProcessConfig YAMLs.

Library modules (called by the entry point):
    convert_params_for_superpsa    - DSL fits + Qst + density -> SuperPSA Adsorbents.csv format.
    generate_process_config        - Template -> ProcessConfig_{PSA,VSA}.yaml.

Legacy:
    curve_cache                    - Breakthrough-curve cache tables (BKT, to be removed).
"""
