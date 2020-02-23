

class Stim:

    def __init__(self, name, id, size):
        self.name = name
        self.id = id
        self.size = [int(x) for x in size.split(',')]