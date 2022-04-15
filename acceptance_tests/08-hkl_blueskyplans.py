###  This makes sure scanning with hkl motors is working correctly (not the caluculations, but the ophyd part).  Relative motion plans in pre-assembled plans break the most frequently.

difference_tol =  0.001  #this number is affected by iterations so different from beamline to beamline



####  Set up to make a UB - just using someting for 853eV, which is why energy is hard-coded.
sample_type = 'LSNO',
sample_composition = '1/3-200-s3'

lattice = Lattice(a = 3.84, b = 3.84, c = 12.65, alpha = 90, beta = 90, gamma = 90)
tardis.calc.new_sample('LSNO',lattice=lattice)

tardis.calc['gamma'].limits = (-5,95)

## UB matrix definition - nominal orientation
if lattice is not None:
    tardis.lattice.put(lattice)
else:
    #TODO put our generic lattice to make qx, qy, qz
    pass
# Standard fake UB
r1 = tardis.calc.sample.add_reflection(0, 0, 1, position=tardis.calc.Position(theta=10.0, mu=0.0, chi=0.0, phi=0.0, delta=20.0, gamma=0.0))
r2 = tardis.calc.sample.add_reflection(1, 0, 0, position=tardis.calc.Position(theta=100.0, mu=0.0, chi=0.0, phi=0.0, delta=20.0, gamma=0.0))
#tardis.calc.sample.compute_UB(r1, r2)

r3_p = tardis.calc.sample.add_reflection(1, 1, 0, position=tardis.calc.Position(theta=62.8797, mu=0.0, chi=0.0, phi=0.0, delta=127.1217, gamma=3.3632)) #   
r4_p = tardis.calc.sample.add_reflection(1, 1, 2, position=tardis.calc.Position(theta=99.4303, mu=0.0, chi=0.0, phi=0.0, delta=153.9771, gamma=2.8022)) #
tardis.calc.sample.compute_UB(r3_p, r4_p)

tardis.calc.energy = (853)/1000  #grt3 focus with ni target at 853 #05-15-21 

# TESTING STARTS HERE
myhkl = (0.1, 0.1, 0.00)
hh, kk, ll = myhkl

print('Test 1 - absolute motion')
ideal_pos = tardis.forward(myhkl)
RE(mv(tardis.h, hh, tardis.k, kk, tardis.l, ll))
tardis_pos = tardis.real_position

for i_pos, t_pos, t in zip(ideal_pos, tardis_pos, tardis.real_positioners):
    pass_fail = ''
    d_pos = abs(i_pos - t_pos)
    if d_pos > difference_tol:  #this number is affected by iterations so different from beamline to beamline
        pass_fail = '**FAILED**'
    print(f'\t{t.name} real motor difference = {d_pos:.6f}  {pass_fail}')

print('Test 2 - relative motion more than 1')
rel_hkl = (-0.05, 0, 0.05)
dh, dk, dl = rel_hkl
ideal_pos = tardis.forward(hh+dh, kk+dk, ll+dl)
RE(mvr(tardis.h, dh, tardis.k, dk, tardis.l, dl))
tardis_pos = tardis.real_position

for i_pos, t_pos, t in zip(ideal_pos, tardis_pos, tardis.real_positioners):
    pass_fail = ''
    d_pos = abs(i_pos - t_pos)
    if d_pos > difference_tol:  #this number is affected by iterations so different from beamline to beamline
        pass_fail = '**FAILED**'
    print(f'\t{t.name} real motor difference = {d_pos:.6f}  {pass_fail}')
RE(mv(tardis.h, hh, tardis.k, kk, tardis.l, ll))


print('Test 3 - relative motion 1st only')
rel_hkl = (-0.05, 0, 0.)
dh, dk, dl = rel_hkl
ideal_pos = tardis.forward(hh+dh, kk+dk, ll+dl)
RE(mvr(tardis.h, dh, tardis.k, dk, tardis.l, dl))
tardis_pos = tardis.real_position

for i_pos, t_pos, t in zip(ideal_pos, tardis_pos, tardis.real_positioners):
    pass_fail = ''
    d_pos = abs(i_pos - t_pos)
    if d_pos > difference_tol:  #this number is affected by iterations so different from beamline to beamline
        pass_fail = '**FAILED**'
    print(f'\t{t.name} real motor difference = {d_pos:.6f}  {pass_fail}')
RE(mv(tardis.h, hh, tardis.k, kk, tardis.l, ll))



print('Test 4 - relative motion more 3rd only')
rel_hkl = (0, 0, 0.05)
dh, dk, dl = rel_hkl
print(tardis.position)
print(hh, kk, ll , "to", hh+dh, kk+dk, ll+dl)
ideal_pos = tardis.forward(hh+dh, kk+dk, ll+dl)
RE(mvr(tardis.h, dh, tardis.k, dk, tardis.l, dl))
tardis_pos = tardis.real_position
print(ideal_pos)
print(tardis_pos)

for i_pos, t_pos, t in zip(ideal_pos, tardis_pos, tardis.real_positioners):
    pass_fail = ''
    d_pos = abs(i_pos - t_pos)
    if d_pos > difference_tol:  #this number is affected by iterations so different from beamline to beamline
        pass_fail = '**FAILED**'
    print(f'\t{t.name} real motor difference = {d_pos:.6f}  {pass_fail}')
RE(mv(tardis.h, hh, tardis.k, kk, tardis.l, ll))
