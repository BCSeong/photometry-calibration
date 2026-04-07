#!/usr/bin/env python3
"""
Photometry Calibration - Main Entry Point

Usage:
    python main.py auto [options]           # Auto calibration (CLI only)
    python main.py auto --interactive       # Auto calibration (interactive prompts)
    python main.py manual [options]         # Manual calibration (CLI only)
    python main.py manual --interactive     # Manual calibration (interactive prompts)
"""

import argparse
import sys


def _prompt_if_missing(value, prompt_text, type_fn=str, default=None):
    """CLI 인자가 없으면 interactive prompt로 보충."""
    if value is not None:
        return value
    raw = input(prompt_text).strip()
    if not raw:
        return default
    try:
        return type_fn(raw)
    except ValueError:
        if default is not None:
            print(f"Invalid input. Using default: {default}")
            return default
        raise


def cmd_auto(args):
    """Auto calibration 서브커맨드."""
    from photometry_calibration.auto_calibration import run_auto_calibration

    if args.interactive:
        image_pattern = _prompt_if_missing(
            args.image_pattern,
            "Enter image directory or pattern (e.g. L2 or L2/*.bmp): ",
            default="./*.bmp")
        sphere_diameter = _prompt_if_missing(
            args.sphere_diameter,
            "Enter sphere diameter (mm): ",
            type_fn=float, default=3.0)
        pixel_resolution = _prompt_if_missing(
            args.pixel_resolution,
            "Enter pixel resolution (mm/px): ",
            type_fn=float, default=0.01)
        num_spheres = _prompt_if_missing(
            args.num_spheres,
            "Enter expected number of spheres: ",
            type_fn=int, default=1)
        remap_dir = _prompt_if_missing(
            args.remap_dir,
            "Enter remap directory path: ")
        highlight_method = _prompt_if_missing(
            args.highlight_method,
            "Highlight position method? (Enter = centroid, 'ring' or 'r' = ring): ",
            default='centroid')
        save_dir = _prompt_if_missing(
            args.save_dir,
            "Enter save directory (or press Enter for default): ",
            default='./output_auto_calibration')
    else:
        image_pattern = args.image_pattern
        sphere_diameter = args.sphere_diameter
        pixel_resolution = args.pixel_resolution
        num_spheres = args.num_spheres
        remap_dir = args.remap_dir
        highlight_method = args.highlight_method or 'centroid'
        save_dir = args.save_dir or './output_auto_calibration'

        # CLI 모드에서 필수 인자 검증
        missing = []
        if not image_pattern:
            missing.append('--image_pattern')
        if sphere_diameter is None:
            missing.append('--sphere_diameter')
        if pixel_resolution is None:
            missing.append('--pixel_resolution')
        if num_spheres is None:
            missing.append('--num_spheres')
        if not remap_dir:
            missing.append('--remap_dir')
        if missing:
            print(f"Error: missing required arguments: {', '.join(missing)}")
            print("Use --interactive flag to enter values interactively.")
            sys.exit(1)

    # highlight_method 정규화
    if highlight_method in ('centroid', 'c'):
        highlight_method = 'centroid'
    else:
        highlight_method = 'ring'

    run_auto_calibration(
        image_pattern=image_pattern,
        sphere_diameter=sphere_diameter,
        pixel_resolution=pixel_resolution,
        num_spheres_expected=num_spheres,
        remap_dir=remap_dir,
        save_base=save_dir,
        highlight_method=highlight_method,
    )


def cmd_manual(args):
    """Manual calibration 서브커맨드."""
    from photometry_calibration.manual_calibration import run_manual_calibration

    if args.interactive:
        image_pattern = _prompt_if_missing(
            args.image_pattern,
            "Enter image directory or pattern (e.g. L2 or L2/*.bmp): ",
            default="L2")
        sphere_diameter = _prompt_if_missing(
            args.sphere_diameter,
            "Enter sphere diameter (mm): ",
            type_fn=float, default=3.0)
        pixel_resolution = _prompt_if_missing(
            args.pixel_resolution,
            "Enter pixel resolution (mm/px): ",
            type_fn=float, default=0.01)
        num_spheres = _prompt_if_missing(
            args.num_spheres,
            "Enter number of spheres per image (1 for single sphere): ",
            type_fn=int, default=1)

        mode_input = input(
            "Draw highlight region manually? (Enter = manual, 'auto' or 'a' = auto): "
        ).strip().lower()
        auto_mode = mode_input in ('auto', 'a')

        highlight_method = _prompt_if_missing(
            args.highlight_method,
            "Highlight position method? (Enter = centroid, 'ring' or 'r' = ring): ",
            default='centroid')
        remap_dir = _prompt_if_missing(
            args.remap_dir,
            "Enter remap directory path (or press Enter to skip): ")
        save_dir = _prompt_if_missing(
            args.save_dir,
            "Enter save directory (or press Enter for default): ",
            default='./output_calibration_results')
    else:
        image_pattern = args.image_pattern
        sphere_diameter = args.sphere_diameter
        pixel_resolution = args.pixel_resolution
        num_spheres = args.num_spheres or 1
        auto_mode = args.auto_highlight or False
        highlight_method = args.highlight_method or 'centroid'
        remap_dir = args.remap_dir
        save_dir = args.save_dir or './output_calibration_results'

        missing = []
        if not image_pattern:
            missing.append('--image_pattern')
        if sphere_diameter is None:
            missing.append('--sphere_diameter')
        if pixel_resolution is None:
            missing.append('--pixel_resolution')
        if missing:
            print(f"Error: missing required arguments: {', '.join(missing)}")
            print("Use --interactive flag to enter values interactively.")
            sys.exit(1)

    # highlight_method 정규화
    if highlight_method in ('ring', 'r'):
        highlight_method = 'ring'
    else:
        highlight_method = 'centroid'

    run_manual_calibration(
        image_pattern=image_pattern,
        sphere_diameter=sphere_diameter,
        pixel_resolution=pixel_resolution,
        num_spheres=num_spheres,
        auto_mode=auto_mode,
        highlight_method=highlight_method,
        remap_dir=remap_dir,
        save_base=save_dir,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Photometry Calibration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py auto --interactive
  python main.py auto --image_pattern L2 --sphere_diameter 4.0 --pixel_resolution 0.01 --num_spheres 7 --remap_dir ./remap
  python main.py manual --interactive
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Calibration mode")

    # --- auto subcommand ---
    auto_parser = subparsers.add_parser("auto", help="Auto calibration (sphere auto-detection)")
    auto_parser.add_argument("--interactive", "-i", action="store_true",
                             help="Interactive mode: prompt for missing parameters")
    auto_parser.add_argument("--image_pattern", type=str, default=None,
                             help="Image directory or glob pattern")
    auto_parser.add_argument("--sphere_diameter", type=float, default=None,
                             help="Sphere diameter (mm)")
    auto_parser.add_argument("--pixel_resolution", type=float, default=None,
                             help="Pixel resolution (mm/px)")
    auto_parser.add_argument("--num_spheres", type=int, default=None,
                             help="Expected number of spheres")
    auto_parser.add_argument("--remap_dir", type=str, default=None,
                             help="Remap map directory path")
    auto_parser.add_argument("--highlight_method", type=str, default=None,
                             choices=['centroid', 'ring'],
                             help="Highlight position method (default: centroid)")
    auto_parser.add_argument("--save_dir", type=str, default=None,
                             help="Output directory for results")
    auto_parser.set_defaults(func=cmd_auto)

    # --- manual subcommand ---
    manual_parser = subparsers.add_parser("manual", help="Manual calibration (matplotlib GUI)")
    manual_parser.add_argument("--interactive", "-i", action="store_true",
                               help="Interactive mode: prompt for missing parameters")
    manual_parser.add_argument("--image_pattern", type=str, default=None,
                               help="Image directory or glob pattern")
    manual_parser.add_argument("--sphere_diameter", type=float, default=None,
                               help="Sphere diameter (mm)")
    manual_parser.add_argument("--pixel_resolution", type=float, default=None,
                               help="Pixel resolution (mm/px)")
    manual_parser.add_argument("--num_spheres", type=int, default=None,
                               help="Number of spheres per image")
    manual_parser.add_argument("--auto_highlight", action="store_true",
                               help="Auto-detect highlight region (default: manual)")
    manual_parser.add_argument("--highlight_method", type=str, default=None,
                               choices=['centroid', 'ring'],
                               help="Highlight position method (default: centroid)")
    manual_parser.add_argument("--remap_dir", type=str, default=None,
                               help="Remap map directory path (optional)")
    manual_parser.add_argument("--save_dir", type=str, default=None,
                               help="Output directory for results")
    manual_parser.set_defaults(func=cmd_manual)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
