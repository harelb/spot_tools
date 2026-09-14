"""Observation-based placement verification shared by hardware backends.

This checks a newly observed class near the reviewed destination. It does not
claim instance identity from class or proximity, and uses no simulator truth.
"""
import math
import time


def observed_candidates(response, *, episode, after_ns, target, requested_at, radius=.20):
    capture=response.get('camera_model',{}).get('live_observation',{})
    if (capture.get('kind')!='live' or capture.get('episode_id')!=episode
            or len(capture.get('calibration_sha256',''))!=64
            or int(capture.get('timestamp_ns',-1))<=after_ns
            or not math.isfinite(capture.get('received_at',0))
            or capture.get('received_at',0)<requested_at):
        raise RuntimeError('Placement verification received stale or unregistered evidence')
    stamp=str(capture['timestamp_ns'])
    matches=[]
    for detection in response.get('detections',[]):
        source=detection.get('observation_source',{})
        if (source.get('kind')!='live' or source.get('episode_id')!=episode
                or source.get('timestamp_ns')!=stamp
                or source.get('calibration_sha256')!=capture['calibration_sha256']):
            raise RuntimeError('Placement detection does not match its calibrated capture')
        position=detection.get('position',[])
        if len(position)!=3 or not all(math.isfinite(v) for v in position):
            raise RuntimeError('Placement detection has invalid geometry')
        if math.isfinite(detection.get('confidence',0)) and detection.get('confidence',0)>=.5 and math.dist(position,target)<=radius:
            matches.append(dict(position=position,confidence=detection['confidence'],
                                observation_source=source))
    return matches,capture


class PlacementVerifier:
    def __init__(self, robot, search, episode, stream_id, cancelled=lambda:False):
        self.robot,self.search,self.episode,self.cancelled=robot,search,episode,cancelled
        self.stream_id=stream_id

    def state(self):
        from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME, BODY_FRAME_NAME
        state=self.robot.state_client.get_robot_state()
        stamp=state.kinematic_state.acquisition_timestamp
        pose=get_a_tform_b(state.kinematic_state.transforms_snapshot,VISION_FRAME_NAME,BODY_FRAME_NAME)
        return state,stamp.seconds*10**9+stamp.nanos,pose

    def query(self, object_class, target, after_ns):
        if self.cancelled():raise RuntimeError('Placement verification cancelled')
        requested=time.time()
        response=self.search(dict(query=object_class,predicate='exists',tier='live',
            live_stream_id=self.stream_id,required_confidence=.5,models={},
            settings={'after_timestamp_ns':after_ns}))
        if self.cancelled():raise RuntimeError('Placement verification cancelled')
        return observed_candidates(response,episode=self.episode,after_ns=after_ns,
                                   target=target,requested_at=requested)

    def prepare(self, object_class, position):
        if position is None:raise RuntimeError('Verified placement requires an approved point')
        state,stamp,pose=self.state()
        if not state.manipulator_state.is_gripper_holding_item:
            raise RuntimeError('Placement verification requires observed holding state')
        target=list(pose.transform_point(*position))
        matches,capture=self.query(object_class,target,stamp)
        if matches:raise RuntimeError('Destination already contains this class; placement would be ambiguous')
        return dict(target=target,before=capture,object_class=object_class)

    def verify(self, context):
        state,stamp,_=self.state()
        if state.manipulator_state.is_gripper_holding_item:
            raise RuntimeError('Placement release did not empty the gripper')
        matches,capture=self.query(context['object_class'],context['target'],stamp)
        if len(matches)!=1:
            raise RuntimeError('Placed object was not uniquely reobserved near the approved destination')
        return dict(verified=True,target=context['target'],before=context['before'],
                    after=capture,detection=matches[0],
                    claim='new class observation near destination; not instance-identity proof')
