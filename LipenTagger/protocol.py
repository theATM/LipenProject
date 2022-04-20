from collections import OrderedDict

labels = dict(
     triangle = 0,
     ruler    = 1,
     gum      = 2,
     pencil   = 3,
     pen      = 4,
     none     = 5,
     wrong    = 6
)


sublabels = dict(
    sub_triangle = dict(
        t45 = 0,
        t60 = 1
    ),
    sub_ruler=dict(
        r15=0,
        r30=1
    ),
    sub_gum=None,
    sub_pencil=dict(
        normal_empty=0,
        small_empty=1,
        normal_gum=2,
        small_gum=3
    ),
    sub_pen=dict(
        normal=0,
        cap=1
    ),
    sub_none=dict(
        thing=0,
        background=1
    ),
    sub_wrong=None,
)

extralabels = dict(
    broken = 2,
    blurred = 4,
    dark = 8,
    lamp = 16,
    hard = 32,
    normal=0
)

