from bluesky.plans import rel_scan
#from bluesky.callbacks import LiveTable, LivePlot


assert sclr.connected
assert cam_dif.connected

#subs = [LiveTable(['eta', 'sclr_ch3', 'cam_dif_stats5_total']),
#        LivePlot('cam_dif_stats5_total', 'eta')]

RE(rel_scan([sclr, cam_dif], theta, -.1, .1, 5))
