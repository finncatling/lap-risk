import time


class Timer:
    """Convenience wrapper on timer function."""

    def __init__(self, start_on_instantiation=True):
        if start_on_instantiation:
            self.start = time.time()
        else:
            self.start = None
        self.stopped = False

    def elapsed(self, raw=False):
        """Displays elapsed time. If raw=True, returns datetime, else
            returns ready-formatted string."""
        if self.start:
            if self.stopped:
                diff = self.stopped - self.start
            else:
                diff = time.time() - self.start

            if raw:
                return diff
            else:
                return '(' + str(round(diff / 60, 2)) + ' minutes elapsed)'

    def start(self):
        """Starts timer, if it isn't already started."""
        if self.start is None:
            self.start = time.time()
        else:
            print('Timer is already started.')

    def stop(self):
        """Stops timer."""
        if self.stopped:
            print('Timer is already stopped.')
        else:
            self.stopped = time.time()


class Reporter:
    """Give timestamped progress updates to end user."""

    def __init__(self):
        self.timer = Timer()

    def report(self, message, leading_newline=True):
        """Prints message for end user.

        Args:
            message (str): Message
            leading_newline (bool): If True, starts message with a new line.

        Returns:
            None
        """
        if leading_newline:
            print('\n', message, ' ', self.timer.elapsed(), sep='')
        else:
            print(message, self.timer.elapsed())
