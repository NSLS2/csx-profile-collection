from bluesky.plans import rel_scan
#from bluesky.callbacks import LiveTable, LivePlot


assert sclr.connected
assert dif_beam.connected

#subs = [LiveTable(['eta', 'sclr_ch3', 'dif_beam_stats5_total']),
#        LivePlot('dif_beam_stats5_total', 'eta')]

RE(rel_scan([sclr, dif_beam], theta, -.1, .1, 5))
