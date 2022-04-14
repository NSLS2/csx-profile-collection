dif_beam.stats5.total.kind = 'hinted'
sclr.channels.chan1.kind = 'hinted'
sclr.channels.chan2.kind = 'hinted'


uids = RE(count([dif_beam, sclr], num=3))
assert uids
headers = db[uids]
print(headers)

