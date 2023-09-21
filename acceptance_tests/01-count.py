from bluesky.plans import count
#from bluesky.callbacks import LiveTable, LivePlot


assert cam_dif.connected
assert sclr.connected
cam_dif.stats5.total.kind = 'hinted'
sclr.channels.chan1.kind = 'hinted'
sclr.channels.chan2.kind = 'hinted'


RE(count([cam_dif, sclr], num=3))
