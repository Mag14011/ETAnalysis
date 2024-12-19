import sys

def derrida_main():
    """Entry point for Derrida analysis command line interface."""
    from .DerridaCurrentAnalysis import main as derrida_analysis_main
    try:
        sys.exit(derrida_analysis_main())
    except Exception as e:
        print(f"Error running Derrida analysis: {str(e)}", file=sys.stderr)
        sys.exit(1)

def flux_main():
    """Entry point for flux analysis command line interface."""
    from .FluxCurrentAnalysis import main as flux_analysis_main
    try:
        sys.exit(flux_analysis_main())
    except Exception as e:
        print(f"Error running flux analysis: {str(e)}", file=sys.stderr)
        sys.exit(1)

def cooperativity_main():
    """Entry point for heme cooperativity analysis command line interface."""
    from .AnalyzeHemeCooperativity import main as cooperativity_analysis_main
    try:
        return cooperativity_analysis_main()
    except Exception as e:
        print(f"Error running cooperativity analysis: {str(e)}", file=sys.stderr)
        sys.exit(1)

def parameter_explorer_main():
    """Entry point for parameter exploration command line interface."""
    from .parameter_exploration import main as parameter_exploration_main
    try:
        return parameter_exploration_main()
    except Exception as e:
        print(f"Error running parameter exploration: {str(e)}", file=sys.stderr)
        sys.exit(1)

def parameter_visualization_main():
    """Entry point for parameter visualization command line interface."""
    from .parameter_visualization import main as parameter_visualization_main
    try:
        return parameter_visualization_main()
    except Exception as e:
        print(f"Error running parameter visualization: {str(e)}", file=sys.stderr)
        sys.exit(1)

def redox_titration_main():
    """Entry point for redox titration analysis command line interface."""
    from .redox_titration import main as redox_titration_main
    try:
        return redox_titration_main()
    except Exception as e:
        print(f"Error running redox titration analysis: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    print("This module is not meant to be run directly.")
    print("Please use one of the following commands after installing the package:")
    print("  et-derrida    - Run Derrida analysis")
    print("  et-flux       - Run flux analysis")
    print("  et-coop       - Run cooperativity analysis")
    print("  et-explore    - Run parameter exploration")
    print("  et-visualize  - Run parameter visualization")
    print("  et-redox      - Run redox titration analysis")
    sys.exit(1)
