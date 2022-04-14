from bluesky.plans import rel_scan
#from bluesky.callbacks import LiveTable, LivePlot


assert sclr.connected

#subs = [LiveTable(['eta', 'sclr_ch3']), LivePlot('sclr_ch3', 'eta')]
RE(rel_scan([sclr], theta, -.1, .1, 3))
