from bluesky.plans import count
#from bluesky.callbacks import LiveTable, LivePlot


assert dif_beam.connected
assert sclr.connected
dif_beam.stats5.total.kind = 'hinted'
sclr.channels.chan1.kind = 'hinted'
sclr.channels.chan2.kind = 'hinted'


RE(count([dif_beam, sclr], num=3))
