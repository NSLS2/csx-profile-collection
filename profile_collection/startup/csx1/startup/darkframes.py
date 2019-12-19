import functools

import bluesky.plan_stubs as bps

from bluesky_darkframes import DarkFramePreprocessor, SnapshotDevice
from ..startup.detectors import fccd
from ..startup.optics import inout


class TemporarilyCustomizedDarkFrameProcessor(DarkFramePreprocessor):
    def disable(self):
        ...

    def enable(self):
        ...




def dark_plan(gain, num_dark_images):
    print('test')
    # Close shutter
    ...
    yield from bps.mv(inout,'In')



    gain_bit_dict = {0:'auto', 1: 'x2', 2: 'x1'} #where the keys are the epics values, but then we added md

    # Set gain.
    ...
    yield from bps.mv(fccd.cam.fcric_gain, gain)

    # Set a reasonable number of images for the dark frame.
    reading = yield from bps.read(fccd.cam.num_images)
    # Structure of reading is
    # {fccd.cam.num_images.name: {'value': ..., 'timestamp': ...}}
    original_num_images = reading[fccd.cam.num_images.name]['value']
    print(f'!!!!!!!!!!!! changing num_images to {num_dark_images}\noriginal_num_images: {original_num_images}')
    yield from bps.mv(fccd.cam.num_images, num_dark_images)

    yield from bps.trigger(fccd, group='darkframe-trigger')
    yield from bps.wait('darkframe-trigger')

    snapshot = SnapshotDevice(fccd)

    if gain == 0:
        yield from bps.mv(fccd.fccd1.capture_bgnd, 1)
    # Set num_images back to where it was.
    print(f'!!!!!!!!!!!! changing num_images to {original_num_images}')
    yield from bps.mv(fccd.cam.num_images, original_num_images)

    # Open shutter
    yield from bps.mv(inout,'Out')

    # We need to wait a bit until the original number of frames is
    # picked up (?)
    yield from bps.sleep(2)

    #Im not sure that this is the best place, then it is open /close alot and we should turn the feedback on first

    return snapshot


dark_frame_preprocessors = {}

#get state of gain before darks are taken so we can return it to the state the user was using.
gain_state = fccd.cam.fcric_gain.value
inout_sate = inout.status.value


#for gain in (1, 2, 8):
#for gain in (0, 1, 2):
for gain in (0,):
    # 1, 2, and 8 are the multipliers.  auto = 8,
    #gain_bit_dict = {0:'auto', 1: 'x2', 2: 'x1'} #where the keys are the epics values
    NUM_DARK_IMAGES = 10
    this_dark_plan = functools.partial(dark_plan, gain, NUM_DARK_IMAGES)

    dark_frame_preprocessor = DarkFramePreprocessor(
        dark_plan=this_dark_plan, detector=fccd, max_age=600)
    dark_frame_preprocessors[gain] = dark_frame_preprocessor


# yield from bps.mv(fccd.cam.fcric_gain, gain_state)
# yield from bps.mv(inout,inout_state)
