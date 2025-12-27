"""
Main entry point for Albert robot MPC simulation.

This script sets up and runs the complete simulation, including:
- Environment initialization
- MPC controller setup
- Simulation execution
- Results visualization
"""

import numpy as np
import argparse
from pathlib import Path
from simulation import AlbertSimulation
from visualization import SimulationPlotter, quick_plot


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Albert robot MPC simulation in bar environment'
    )
    
    # Simulation parameters
    parser.add_argument(
        '--robot-urdf',
        type=str,
        default='path/to/albert.urdf',
        help='Path to robot URDF file'
    )
    parser.add_argument(
        '--dt',
        type=float,
        default=0.01,
        help='Simulation time step (default: 0.01)'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=1000,
        help='Maximum simulation steps (default: 1000)'
    )
    parser.add_argument(
        '--no-render',
        action='store_true',
        help='Disable rendering'
    )
    
    # Initial and target poses
    parser.add_argument(
        '--initial-x',
        type=float,
        default=-3.0,
        help='Initial X position (default: -3.0)'
    )
    parser.add_argument(
        '--initial-y',
        type=float,
        default=-2.0,
        help='Initial Y position (default: -2.0)'
    )
    parser.add_argument(
        '--initial-theta',
        type=float,
        default=0.0,
        help='Initial orientation in radians (default: 0.0)'
    )
    parser.add_argument(
        '--target-x',
        type=float,
        default=0.0,
        help='Target X position (default: 0.0)'
    )
    parser.add_argument(
        '--target-y',
        type=float,
        default=0.0,
        help='Target Y position (default: 0.0)'
    )
    parser.add_argument(
        '--target-theta',
        type=float,
        default=0.0,
        help='Target orientation in radians (default: 0.0)'
    )
    
    # MPC parameters
    parser.add_argument(
        '--mpc-horizon',
        type=int,
        default=20,
        help='MPC prediction horizon (default: 20)'
    )
    parser.add_argument(
        '--mpc-wx',
        type=float,
        default=10.0,
        help='MPC state tracking weight (default: 10.0)'
    )
    parser.add_argument(
        '--mpc-wu',
        type=float,
        default=0.1,
        help='MPC input weight (default: 0.1)'
    )
    parser.add_argument(
        '--max-iter',
        type=int,
        default=10,
        help='Maximum MPC solver iterations (default: 10)'
    )
    parser.add_argument(
        '--no-warm-start',
        action='store_true',
        help='Disable warm start for MPC solver'
    )
    
    # Robot configuration
    parser.add_argument(
        '--facing-direction',
        type=str,
        default='-y',
        choices=['x', 'y', '-x', '-y'],
        help='Robot facing direction (default: -y)'
    )
    
    # Output options
    parser.add_argument(
        '--save-plots',
        type=str,
        default=None,
        help='Directory to save plots (default: None, shows plots)'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip plotting results'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.1,
        help='Distance tolerance for reaching target (default: 0.1)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Print configuration
    if not args.quiet:
        print("\n" + "=" * 70)
        print("ALBERT ROBOT MPC SIMULATION")
        print("=" * 70)
        print("\nConfiguration:")
        print(f"  Robot URDF:          {args.robot_urdf}")
        print(f"  Time step:           {args.dt} s")
        print(f"  Max steps:           {args.max_steps}")
        print(f"  Render:              {not args.no_render}")
        print(f"\nInitial pose:          [{args.initial_x:.2f}, {args.initial_y:.2f}, {args.initial_theta:.2f}]")
        print(f"Target pose:           [{args.target_x:.2f}, {args.target_y:.2f}, {args.target_theta:.2f}]")
        print(f"\nMPC horizon:           {args.mpc_horizon}")
        print(f"MPC weights (wx, wu):  ({args.mpc_wx}, {args.mpc_wu})")
        print(f"Solver max iter:       {args.max_iter}")
        print(f"Warm start:            {not args.no_warm_start}")
        print(f"\nFacing direction:      {args.facing_direction}")
        print("=" * 70 + "\n")
    
    # Create initial and target poses
    initial_pose = np.array([args.initial_x, args.initial_y, args.initial_theta])
    target_pose = np.array([args.target_x, args.target_y, args.target_theta])
    
    # Initialize simulation
    try:
        sim = AlbertSimulation(
            robot_urdf=args.robot_urdf,
            initial_pose=initial_pose,
            target_pose=target_pose,
            dt=args.dt,
            mpc_horizon=args.mpc_horizon,
            mpc_weights=(args.mpc_wx, args.mpc_wu),
            facing_direction=args.facing_direction,
            render=not args.no_render,
            max_iter=args.max_iter,
            warm_start=not args.no_warm_start
        )
    except Exception as e:
        print(f"\n✗ Failed to initialize simulation: {e}")
        return 1
    
    # Run simulation
    try:
        history = sim.run(
            max_steps=args.max_steps,
            tolerance=args.tolerance,
            verbose=not args.quiet
        )
    except KeyboardInterrupt:
        print("\n\n✗ Simulation interrupted by user")
        sim.close()
        return 1
    except Exception as e:
        print(f"\n✗ Simulation failed: {e}")
        sim.close()
        return 1
    
    # Close simulation
    sim.close()
    
    # Plot results
    if not args.no_plot:
        try:
            plotter = SimulationPlotter()
            
            # Determine save paths
            if args.save_plots:
                save_dir = Path(args.save_plots)
                save_dir.mkdir(parents=True, exist_ok=True)
                
                all_plots_path = save_dir / "simulation_results.png"
                trajectory_path = save_dir / "trajectory.png"
                control_path = save_dir / "control_analysis.png"
                
                print(f"\nSaving plots to {save_dir}...")
                plotter.plot_all(history, save_path=str(all_plots_path))
                plotter.plot_trajectory_only(history, save_path=str(trajectory_path))
                plotter.plot_control_analysis(history, save_path=str(control_path))
            else:
                # Show plots interactively
                quick_plot(history)
            
            # Print statistics
            if not args.quiet:
                plotter.print_statistics(history)
        
        except Exception as e:
            print(f"\n✗ Plotting failed: {e}")
            return 1
    
    print("\n✓ Simulation completed successfully!\n")
    return 0


if __name__ == "__main__":
    exit(main())
