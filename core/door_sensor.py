"""A motionless open door is not evidence of a closed door.

Only an external door contact may gate counting; default is to analyse all frames.
"""


class DoorSensor:
    def __init__(self, read_closed=None):
        self.read_closed = read_closed

    def is_doors_closed(self, frame=None):
        return bool(self.read_closed()) if self.read_closed is not None else False
