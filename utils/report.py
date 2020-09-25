import time


class Timer:
    """Convenience wrapper on timer function."""

    def __init__(self, start_on_instantiation: bool = True):
        if start_on_instantiation:
            self.start_time = time.time()
        else:
            self.start_time = None
        self.stopped = False

    def elapsed(self, raw: bool = False):
        """Displays elapsed time. If raw=True, returns datetime, else
            returns ready-formatted string."""
        if self.start_time:
            if self.stopped:
                diff = self.stopped - self.start_time
            else:
                diff = time.time() - self.start_time

            if raw:
                return diff
            else:
                return "(" + str(round(diff / 60, 2)) + " minutes elapsed)"

    def start(self):
        """Starts timer, if it isn't already started."""
        if self.start_time is None:
            self.start_time = time.time()
        else:
            print("Timer is already started.")

    def stop(self):
        """Stops timer."""
        if self.stopped:
            print("Timer is already stopped.")
        else:
            self.stopped = time.time()


class Reporter:
    """Give timestamped progress updates to end user."""

    def __init__(self):
        self.timer = Timer()

    def report(
        self,
        message: str,
        leading_newline: bool = False,
        trailing_newline: bool = False,
    ):
        """Prints message for end user.

        Args:
            message: Message
            leading_newline: If True, starts message with a new line
            trailing_newline: If True, end message with a new line
        """
        message = f"{message} {self.timer.elapsed()}"
        if leading_newline:
            message = f"\n{message}"
        if trailing_newline:
            message = f"{message}\n"
        print(message)

    def title(self, message: str, show_time=False, box_width: int = 72):
        """Print message in title box."""
        box_edge = "#" * box_width
        text_width = int(box_width - 4)
        char_i = 0
        if show_time:
            message = f"{message} {self.timer.elapsed()}"
        print(f"\n{box_edge}")
        while char_i < len(message):
            fragment = message[char_i: char_i + text_width]
            extra_spaces = " " * int(text_width - len(fragment))
            print(f"# {fragment}{extra_spaces} #")
            char_i += text_width
        print(f"{box_edge}\n")

    def first(self, message: str):
        """Convenience wrapper for first message."""
        self.report(message, leading_newline=True)

    def last(self, message: str):
        """Convenience wrapper for first message."""
        self.report(message, trailing_newline=True)
