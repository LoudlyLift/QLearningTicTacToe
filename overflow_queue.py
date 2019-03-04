import random

class OverflowQueue():
    """Tracks the most recent items that have been added to
    """
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

    def addOne(self, item):
        assert(len(self.buffer) <= self.buffer_size)
        if (len(self.buffer) == self.buffer_size):
            self.buffer[0:1] = []
        self.buffer.append(item)

    def addMany(self, items):
        assert(len(self.buffer) <= self.buffer_size)
        assert(len(items) <= self.buffer_size)
        overflow = len(self.buffer) + len(items) - self.buffer_size
        if overflow > 0:
            self.buffer[0:overflow] = []
        self.buffer.extend(items)

    def sample(self, max):
        assert(len(self.buffer) <= self.buffer_size)
        assert(max >= 0)
        max = min(len(self.buffer), max)
        return random.sample(self.buffer, max)

    def __len__(self):
        return len(self.buffer)
