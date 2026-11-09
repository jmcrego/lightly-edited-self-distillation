#!/usr/bin/env python3
"""Print a pending Slurm job's position in its partition's priority queue."""

import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", nargs="?", default="1847935", help="Job ID (default: 1847935)")
    parser.add_argument("--partition", default="gpu_p6", help="Partition (default: gpu_p6)")
    args = parser.parse_args()
    try:
        result = subprocess.run(
            ["squeue", "-p", args.partition, "-t", "PD", "--sort=-p", "-h", "-o", "%i"],
            check=True, capture_output=True, text=True,
        )
    except FileNotFoundError:
        parser.exit(1, "Error: squeue is unavailable; run this script on the Slurm cluster.\n")
    except subprocess.CalledProcessError as error:
        parser.exit(1, f"Error: squeue failed: {error.stderr.strip()}\n")
    for position, job in enumerate(result.stdout.splitlines(), 1):
        if job.strip() == args.job:
            print(f"Position in partition: {position}")
            return
    parser.exit(1, f"Job {args.job} is not pending in partition {args.partition}.\n")


if __name__ == "__main__":
    main()
