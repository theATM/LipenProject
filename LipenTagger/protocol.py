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
    sub_ruler=None,
    sub_gum=None,
    sub_pencil=dict(
        normal_empty=0,
        small_empty=1,
        normal_gum=2,
        small_gum=3
    ),
    sub_pen=dict(
        normal=0,
        cap=1,
        backcap=3
    ),
    sub_none=dict(
        thing=0,
        background=1
    ),
    sub_wrong=None,
)

extralabels = dict(
    broken = 1,
    blurred = 2,
    dark = 4,
    lamp = 8,
    hard = 16,
    normal=0
)

