"""Opt-in real Isaac test. Does not issue motion or claim Hydra integration."""
import argparse
import json
import time
from pathlib import Path
from urllib.error import URLError

from spot_executor.isaac_spot import IsaacSpot
from spot_tools_ros.isaac_sensors import SensorClient


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wait',type=float,default=600)
    args=parser.parse_args()
    deadline=time.monotonic()+args.wait
    while True:
        try:
            spot=IsaacSpot();client=SensorClient('http://127.0.0.1:9250')
            if not spot.list_image_sources():raise RuntimeError('sensors not yet populated')
            break
        except (URLError,TimeoutError,RuntimeError):
            if time.monotonic()>deadline:raise
            time.sleep(1)
    rows=[]
    def measure(phase,seconds=10):
        start=time.monotonic();frames=0;first=last=None
        while time.monotonic()-start<seconds:
            result=client.next()
            if result is None:time.sleep(.003);continue
            meta,rgb,depth=result;frames+=1
            assert meta['source']=='front_zed_color_image'
            assert rgb.shape[:2]==depth.shape and (depth>0).any()
            if first is None:first=meta['timestamp_ns']
            last=meta['timestamp_ns']
        elapsed=time.monotonic()-start
        row=dict(phase=phase,frames=frames,elapsed_s=elapsed,fps=frames/elapsed,
            first_stamp_ns=first,last_stamp_ns=last,sources=[s.name for s in spot.list_image_sources()])
        rows.append(row);print(json.dumps(row),flush=True)
    measure('front_zed_only')
    assert rows[-1]['sources']==['front_zed_color_image']
    with spot.pick_camera_session():
        response,rgb=spot.get_image_RGB('hand_color_image')
        assert response.source.name=='hand_color_image' and rgb.size>0
        measure('grasp_camera_active')
        assert 'hand_color_image' in rows[-1]['sources']
    measure('grasp_camera_disabled')
    assert rows[-1]['sources']==['front_zed_color_image']
    try:spot.get_image_RGB('hand_color_image')
    except RuntimeError:pass
    else:raise AssertionError('hand camera remained available outside grasp session')
    args.output.write_text(json.dumps(dict(scope='live Isaac camera lifecycle; no motion or Hydra',
        session=spot.transport.session,phases=rows,passed=True),indent=2))


if __name__=='__main__':main()
