class ApplyQueue:
    """
    Class represents the function apply sequence
    """

    def __init__(self):
        self.stack = []

    def add(self, function, args: tuple) -> 'ApplyQueue':
        """
        Adds function and args to queue. args should be (first arg, second arg, ...).
        The zero arg applied to `f` is the result of previous function in the queue.
        """
        self.stack.append((function, args))
        return self

    def apply(self, first_args):
        """
        Sequentially applies functions to the result of previous function and the args stored in queue.
        """
        res = first_args
        for f, args in self.stack:
            res = f(res, *args)
        return res

    def empty(self) -> bool:
        return len(self.stack) == 0
