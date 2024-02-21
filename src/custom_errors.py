

class ClippingError(Exception):
    """ Raised when the amplitude of the wave exceeds the range of its selected output format.
    For this codebase's default output of 32-bit floating point, the range is -1 to 1.
    =====================  ===========  ===========  =============
         WAV format            Min          Max       NumPy dtype
    =====================  ===========  ===========  =============
    32-bit floating-point  -1.0         +1.0         float32
    32-bit PCM             -2147483648  +2147483647  int32
    16-bit PCM             -32768       +32767       int16
    8-bit PCM              0            255          uint8
    =====================  ===========  ===========  ============="""
    def __init__(self, msg):
        super().__init__(msg)