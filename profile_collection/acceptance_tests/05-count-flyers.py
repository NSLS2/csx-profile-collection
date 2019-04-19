from bluesky.plans import count

ct = count([])
ct.flyers = [topoff_inj, diag6_flyer5, diag6_flyer1]


uid, = RE(ct)
assert len(db[uid].descriptors) == 3  # one event stream per flyer
