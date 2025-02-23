import os
import time
import signal
import argparse
from datetime import datetime
from typing import Dict
from tqdm import tqdm


class ProgressMonitor:
    """Monitor multiple progress bars based on log file entries."""

    def __init__(self, log_file: str, total_steps: int, update_interval: float = 0.1):
        """
        Initialize the progress monitor.

        Args:
            log_file: Path to the log file to monitor
            total_steps: Total number of steps for each progress bar
            update_interval: Sleep interval between log file checks in seconds
        """
        self.log_file = log_file
        self.total_steps = total_steps
        self.update_interval = update_interval
        self.progress_bars: Dict[str, tqdm] = {}
        self.running = True

        # Setup signal handlers for graceful exit
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

    def handle_signal(self, signum: int, frame) -> None:
        """Handle termination signals by setting running flag to False."""
        print("\nReceived signal to terminate. Cleaning up...")
        self.running = False

    def create_progress_bar(self, progress_id: str) -> tqdm:
        """Create a new progress bar with consistent formatting."""
        return tqdm(
            total=self.total_steps,
            desc=f"[{datetime.now().strftime('%H:%M:%S')}] {progress_id}",
            position=len(self.progress_bars),
            unit="steps"
        )

    def monitor(self) -> None:
        """Main monitoring loop to read log file and update progress bars."""
        try:
            # Ensure log file exists
            file_exists = False
            while not file_exists:
                file_exists = os.path.isfile(self.log_file)
                if not file_exists:
                    print(f"Error: Log file not found: {self.log_file}")
                    time.sleep(5)

            with open(self.log_file, "r") as log:
                while self.running:
                    line = log.readline()
                    if line:
                        progress_id = line.strip()

                        # Create new progress bar if needed
                        if progress_id not in self.progress_bars:
                            self.progress_bars[progress_id] = self.create_progress_bar(progress_id)

                        # Update progress and check for completion
                        progress_bar = self.progress_bars[progress_id]
                        if progress_bar.n < self.total_steps:
                            progress_bar.update()

                            # Update description with completion marker if done
                            if progress_bar.n >= self.total_steps:
                                timestamp = datetime.now().strftime('%H:%M:%S')
                                progress_bar.set_description(
                                    f"[{timestamp}] ✓ {progress_id} (Complete)"
                                )
                    time.sleep(self.update_interval)

        except Exception as e:
            print(f"\nError: {str(e)}")
        finally:
            # Clean up progress bars
            for bar in self.progress_bars.values():
                bar.close()


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="Monitor progress of multiple tasks using progress bars.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "log_file",
        help="Path to the log file containing progress IDs"
    )
    parser.add_argument(
        "total_steps",
        type=int,
        help="Total number of steps for each progress bar"
    )
    parser.add_argument(
        "-i", "--interval",
        type=float,
        default=0.1,
        help="Update interval in seconds"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.total_steps <= 0:
        parser.error("total_steps must be greater than 0")
    if args.interval <= 0:
        parser.error("interval must be greater than 0")

    return args


def main() -> None:
    """Main entry point for the script."""
    args = parse_arguments()

    try:
        monitor = ProgressMonitor(
            args.log_file,
            args.total_steps,
            args.interval
        )
        monitor.monitor()
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")


if __name__ == "__main__":
    main()