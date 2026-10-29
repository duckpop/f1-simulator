from f1sim.drivers.tracks import widened


def test_widened_leaves_out_tracks_that_dont_fit(tracks):
    assert len(widened(tracks, "normal")) == 2
    wide = widened(tracks, "26")  # seed 0 doesn't fit at 26 m, seed 1 does
    assert len(wide) == 1 and wide[0].name.endswith("-w26")
