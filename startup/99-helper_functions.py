## just adding functions we keep in defs.py so we don't need to keep dragging them around.  
##TODO evaluate ones for csxtools (because also good for db)
##TODO find better place inside of csx dir for these.


## Utility commands typically in startup.py..
bec.disable_baseline()

## Useful shorthands
ct = count
mv = bps.mv
mvr = bps.mvr
aset = bps.abs_set
rset = bps.rel_set
rsc = rel_scan
nap = bps.sleep


## Temporary fix due to faulty olog.. now not necessary anymore
#def olog_patch(string_to_write):
#    try:
#        olog(string_to_write)
#    except:
#        pass


def ct_dark_all_patch(frames=None):
    yield from mv(inout, "In")
    #yield from mv(diag6_pid.enable, 0)  #DIAG6 NEW delete
    yield from ct_dark_all(frames)
    yield from mv(inout, "Out")
    #yield from mv(diag6_pid.enable, 1)   #DIAG6 NEW delete
    yield from bps.sleep(.1)


def pol_L(pol, epu_cal_offset=None, epu_table_number=None):
    """A helper bluesky plan to change vertical and horizontal polarization

    Parameters:
    -----------
    pol : string
        'H' or 'V' for horizontal (epu phase = 0) or vertical (epu phase = 24.6) polarization for CSX's undulator
    epu_cal_offset : float
        offset to tune the precise epu gap for a particular energy using a particular epu calibration table
    epu_table_number : int
        epu calibration table
    """    

    current_E = pgm.energy.setpoint.get()

    if epu_table_number is None:
        if pol == "H":
            epu_table_number = 5
        elif pol == "V":
            epu_table_number = 6
    
    yield from bps.mv(diag6_pid.enable,0) # OFF epu_table & m1a feedback
    if epu_cal_offset is not None:
        yield from bps.mv(epu2.flt.input_offset, epu_cal_offset)#, diag6_pid.setpoint, pid_setpoint) #update calibrations
        
    if pol == 'H':
        print(f'\n\n\tChanging phase to linear horizontal at THIS energy - {current_E:.2f}eV')
        yield from bps.mv(epu2.table, epu_table_number)
        yield from bps.mv(epu2.phase,0)
    elif pol == 'V':
        print(f'\n\n\tChanging phase to linear vertical at THIS energy - {current_E:.2f}eV')
        yield from bps.mv(epu2.table, epu_table_number)
        yield from bps.mv(epu2.phase,24.6)
    
    
    yield from bps.mv(diag6_pid.enable,1) # finepitch feedback ON
    yield from bps.sleep(0.1)

def pol_C(pol, epu_cal_offset=-267, epu_phase=None):
    '''This circular polarization change uses the horizontal polarization table and will force this table.  
    The offset is a best guess.  You should check the values to be certain.
    
    Fe L3
    Cpos = phase = 16.35 
    Cneg = phase = -16.46 -> -16.50 @ 20210806
    '''
    
    if pol == 'pos' or pol == 'neg':
        current_E = pgm.energy.setpoint.get()

        if pol == 'pos' and epu_phase is None:
            epu_phase = 16.35
        elif pol == 'neg' and epu_phase is None:
            epu_phase = -16.46
        
        yield from bps.mv(diag6_pid.enable,0) # OFF epu_table & m1a feedback
        yield from bps.mv(epu2.table, 2)  # forces use of LH polarization table
        yield from bps.mv(epu2.flt.input_offset, epu_cal_offset)
        #TODO add verbose stuff if claudio agrees
        #if pol == 'pos':
        #    print(f'\n\n\tChanging phase to C+ or pos at THIS energy - {current_E:.2f}eV')
        #    
        #    yield from bps.mv(epu2.phase,16.35)
        #elif pol == 'neg':
        #    print(f'\n\n\tChanging phase to C- or neg at THIS energy - {current_E:.2f}eV')
            
        yield from bps.mv(epu2.phase, epu_phase)
        
        yield from bps.mv(diag6_pid.enable,1) # finepitch feedback ON
        yield from bps.sleep(1)
    else:
        print('Allowed arguments for circular polarization are "neg" or "pos"')
        raise


def md_info(default_md = RE.md):
    '''Formatted print of RunEngine metadata or a scan_id start document
    Default behavior prints RE.md.'''


    print('Current default metadata for each scan are:')
    #print(len(default_md))
    for info in default_md:
        val = default_md[info]
        
        #print(info, val)
        print(f'    {info:_<30} : {val}')
    print('\n\n Use \'md_info()\' or \'RE.md\' to inspect again.')

def mvslt3(size=None): #TODO make a better version for slt3.pinhole child    
    """A helper bluesky plan to move to different pinhole diameters for slit 3.

    Parameters:
    -----------
    size : int
        pinhole size in microns.  options are 2000, 50, 20, and 10.  if None, then the current pinhole is returned.

    """
    #x Mtr.OFF = 4.88, y Mtr.OFF = -0.95
    holes = {2000: (  8.800,  0.000),    #TODO eventually have IOC to track these values
               50: (  0.000,  0.000),
               20: (- 8.727,  0.050),
               10: (-17.350, -0.075),} 
    if size is None:
        _xpos = round( slt3.x.read()['slt3_x']['value'], 3)
        _ypos = round( slt3.y.read()['slt3_y']['value'], 3)

        holes_reverse = dict((v, k) for k, v, in holes.items())
        try:
            _size_slt3_pinhole = holes_reverse[(_xpos, _ypos)]
            print(f'{_size_slt3_pinhole}um pinhole at slt3: slt3.x = {_xpos:.4f}, slt3.y = {_ypos:.4f}') 
        except KeyError:
            print(f'Unknown configuration: slt3.x = {_xpos:.4f}, slt3.y = {_ypos:.4f}') 

    else:
        print('Moving to {} um slit 3'.format(size))
        x_pos, y_pos = holes[size]
        yield from bps.mv(slt3.x, x_pos, slt3.y, y_pos)


def wait_for_peaks(pool_interval, timeout, peaks, peaks_fields=None):
    """A helper bluesky plan to wait until the peak fields are calculated wth the specified pooling interval up to the maximum timeout value.

    Parameters:
    -----------
    pool_interval : float
        pooling interval in seconds.
    timeout : float
        maximum time to wait in seconds. The TimeoutError is raised if the time has passed.
    peaks : bluesky.callbacks.best_effort.PeakResults
        the PeakStats object.
    peaks_fields : list or tuple
        the PeakStats fields to wait for.
    """
    if peaks_fields is not None and type(peaks_fields) in [list, tuple]:
        start_time = ttime.monotonic()
        while True:
            if timeout is not None:
                if ttime.monotonic() - start_time > timeout:
                    raise TimeoutError(f"Failed to get peaks calculated within {timeout} seconds")
            all_fields = []
            for field in peaks_fields:
                if getattr(peaks, field):
                    all_fields.append(True)
                else:
                    all_fields.append(False)
            if all(all_fields):
                print(f"Took {ttime.monotonic() - start_time:.6f} seconds to calculate peaks.")
                break
            else:
                yield from bps.sleep(pool_interval)

def _block_beam(block_beam_bit): ##TODO move this to a child of inout
    """Protective inout function that only moves if necessary.
    The EPS will always actuate on bps.mv.

    Parameters:
    ------------
    block_beam_bit : int, bool
    """
    if block_beam_bit: 
        if inout.status.get() == "Not Inserted":
            yield from mv(inout, "In")
    else:
        if inout.status.get() == "Inserted":
            yield from mv(inout, "Out")

def block_beam():
    """Helper plan to move inout to block beam. Aliased to beam_block()."""
    yield from _block_beam(1)

def show_beam():
"""Helper plan to move inout to show beam."""
    yield from _block_beam(0)

# Convenience aliases: Don't need to memorize order of noun/verb
beam_block = block_beam
beam_show = show_beam
