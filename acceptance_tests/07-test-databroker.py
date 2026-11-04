cam_dif.stats5.total.kind = 'hinted'
sclr.channels.chan1.kind = 'hinted'
sclr.channels.chan2.kind = 'hinted'


uids = RE(count([cam_dif, sclr], num=3))
assert uids
flush_tiled()
headers = db[uids]
print(headers)

