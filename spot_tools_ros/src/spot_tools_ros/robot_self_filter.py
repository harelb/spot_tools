"""Exclude known robot geometry from mapping using capture-time proprioception.

Unknown pixels stay unknown (zero depth). This never writes background depth,
and leaves the original sensor observation available for manipulation.
"""
import itertools
import numpy as np
from scipy.spatial.transform import Rotation


def mask_robot_depth(meta,depth,padding=.025):
    if 'robot_self_filter' not in meta:raise ValueError('Capture-time robot geometry is required for mapping')
    info=meta['robot_self_filter']
    if not np.isfinite(padding) or not 0<=padding<=.05:
        raise ValueError('Invalid self-filter padding')
    if not 0<len(info.get('boxes',[]))<=64:
        raise ValueError('Bounded robot geometry is required')
    if info.get('timestamp_ns')!=meta['timestamp_ns']:
        raise ValueError('Self filter state does not match the camera timestamp')
    K=np.asarray(meta['K']);q=np.asarray(meta['orientation_wxyz'])
    camera_rotation=Rotation.from_quat(q[[1,2,3,0]])
    camera_position=np.asarray(meta['position'])
    output=np.array(depth,copy=True);masked=np.zeros(depth.shape,dtype=bool)
    h,w=depth.shape
    for box in info['boxes']:
        low=np.asarray(box['low'],dtype=float)-padding;high=np.asarray(box['high'],dtype=float)+padding
        position=np.asarray(box['position'],dtype=float);q=np.asarray(box['orientation_wxyz'],dtype=float)
        if (low.shape!=(3,) or high.shape!=(3,) or position.shape!=(3,) or q.shape!=(4,)
                or not np.isfinite([*low,*high,*position,*q]).all() or (high<=low).any()
                or max(high-low)>2. or abs(np.linalg.norm(q)-1)>.001):
            raise ValueError('Invalid robot self-filter bound')
        rotation=Rotation.from_quat(q[[1,2,3,0]])
        corners=np.array(list(itertools.product(*zip(low,high))))
        optical=camera_rotation.inv().apply(rotation.apply(corners)+position-camera_position)
        if optical[:,2].max()<=.01:continue
        if optical[:,2].min()<=.01:x0,y0,x1,y1=0,0,w,h
        else:
            uv=optical[:,:2]/optical[:,2,None]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
            x0,y0=np.maximum(0,np.floor(uv.min(axis=0)).astype(int)-1)
            x1,y1=np.minimum([w,h],np.ceil(uv.max(axis=0)).astype(int)+2)
        if x1<=x0 or y1<=y0:continue
        z=depth[y0:y1,x0:x1]
        v,u=np.nonzero(np.isfinite(z)&(z>0))
        if not len(v):continue
        points=np.column_stack(((u+x0-K[0,2])*z[v,u]/K[0,0],
                                (v+y0-K[1,2])*z[v,u]/K[1,1],z[v,u]))
        local=rotation.inv().apply(camera_rotation.apply(points)+camera_position-position)
        inside=np.all((local>=low)&(local<=high),axis=1)
        masked[y0+v[inside],x0+u[inside]]=True
    output[masked]=0
    return output,int(masked.sum())
