

class PenaltyBuilder(object):
    """Returns the Length and Coverage Penalty function for Beam Search.

    Args:
        length_pen (str): option name of length pen

    Attributes:
        has_len_pen (bool): Whether length penalty is None (applying it
            is a no-op). Note that the converse isn't true. Setting alpha
            to 1 should force length penalty to be a no-op.
        length_penalty (callable[[int, float], float]): Calculates
            the length penalty.
    """

    def __init__(self, length_pen):
        self.has_len_pen = not self._pen_is_none(length_pen)
        self.length_penalty = self._length_penalty(length_pen)

    @staticmethod
    def _pen_is_none(pen):
        return pen == "none" or pen is None

    def _length_penalty(self, length_pen):
        if length_pen == "wu":
            return self.length_wu
        elif length_pen == "avg":
            return self.length_average
        elif self._pen_is_none(length_pen):
            return self.length_none
        else:
            raise NotImplementedError("No '{:s}' length penalty.".format(length_pen))

    def length_wu(self, cur_len, alpha=0.0):
        """GNMT length re-ranking score.

        See "Google's Neural Machine Translation System" :cite:`wu2016google`.
        """

        return ((5 + cur_len) / 6.0) ** alpha

    def length_average(self, cur_len, alpha=1.0):
        """Returns the current sequence length."""
        return cur_len**alpha

    def length_none(self, cur_len, alpha=0.0):
        """Returns unmodified scores."""
        return 1.0
