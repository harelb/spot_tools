"""Shared manual/planned ownership below the UI and above hardware clients.

All motion still uses the Spot Tools executor's injected Spot interface. An
operator lease is invalidated on watchdog expiry; reconnect never resumes it.
"""
import math
import threading
import time
import uuid
from bosdyn.client.robot_command import RobotCommandBuilder


class ManualControl:
    def __init__(self, executor, feedback, *, run_id, episode_id, ready=lambda: True,
                 clock=time.time, monotonic=time.monotonic, start_watchdog=True):
        self.executor, self.feedback = executor, feedback
        self.run_id, self.episode_id = run_id, episode_id
        self.ready, self.clock = ready, clock
        self.monotonic=monotonic
        self.lock = threading.RLock()
        self.mode = 'planned'
        self.token = None
        self.sequence = -1
        self.deadline = 0.
        self.error = None
        self.compute_token = None
        self.compute_deadline = 0.
        self.closed = threading.Event()
        self.thread = None
        if start_watchdog:
            self.thread = threading.Thread(target=self._watch, daemon=True)
            self.thread.start()

    def _identity(self, request):
        if request.get('run_id') != self.run_id or request.get('episode_id') != self.episode_id:
            raise ValueError('Stale run or episode')

    def _stop(self):
        # Invalidate ownership even if hardware communication fails.
        self.token = None
        self.deadline = 0.
        self.executor.keep_going = False
        if hasattr(self.executor, "cancel_event"):
            self.executor.cancel_event.set()
        self.feedback.break_out_of_waiting_loop = True
        self.executor.spot_interface.command_client.robot_command(RobotCommandBuilder.stop_command())

    def stop(self, request):
        self._identity(request)
        with self.lock:
            self._stop()
            return self.state()

    def claim(self, request):
        self._identity(request)
        if request.get('mode') not in ('planned', 'manual'):
            raise ValueError('Unknown control mode')
        with self.lock:
            if self.compute_token:
                raise ValueError('Robot is held for perception or motion planning')
            self._stop()
            # Acceptance by the hardware alone is insufficient for a handoff.
            if self.executor.processing_action_sequence:
                raise ValueError('Stopping current skill; wait for terminal feedback before switching')
            self.mode = request['mode']
            self.token = uuid.uuid4().hex
            self.sequence = -1
            self.error = None
            return self.state()

    def reserve_compute(self, request):
        self._identity(request)
        with self.lock:
            if self.compute_token:
                raise ValueError('Another computation holds the robot')
            self._stop()
            if self.executor.processing_action_sequence:
                raise ValueError('Wait for the active skill to acknowledge stopping')
            self.compute_token = uuid.uuid4().hex
            self.compute_deadline = self.monotonic() + 125.
            return {'compute_token': self.compute_token}

    def release_compute(self, request):
        self._identity(request)
        with self.lock:
            if not self.compute_token or request.get('compute_token') != self.compute_token:
                raise ValueError('Stale compute reservation')
            self.compute_token = None
            self.compute_deadline = 0.
            # The previous teleop lease was revoked. Motion needs new review
            # or a new manual claim, even after successful computation.
            return self.state()

    def drive(self, request):
        self._identity(request)
        with self.lock:
            if self.mode != 'manual' or not self.token or request.get('control_token') != self.token:
                raise ValueError(self.error or 'Acquire manual control before driving')
            if self.executor.processing_action_sequence:
                raise ValueError('A skill still owns motion')
            seq = request.get('sequence')
            if type(seq) is not int or seq <= self.sequence:
                raise ValueError('Duplicate or reordered teleop command')
            issued = float(request['issued_at'])
            now = self.clock()
            if not math.isfinite(issued) or not 0 <= now-issued <= .35:
                raise ValueError('Expired teleop command or unsynchronized operator clock')
            velocity = [float(request[k]) for k in ('vx','vy','wz')]
            if not all(map(math.isfinite, velocity)) or math.hypot(*velocity[:2]) > .25 or abs(velocity[2]) > .5:
                raise ValueError('Invalid teleop velocity')
            if not self.ready():
                self.error = 'Fresh state and collision coverage are required'
                self._stop()
                raise ValueError(self.error)
            self.sequence = seq
            self.deadline = self.monotonic() + max(0.,.35-(now-issued))
            try:
                self.executor.spot_interface.set_twist(*velocity)
            except Exception as exc:
                self.error = str(exc)
                self._stop()
                raise
            return {'accepted':True, 'sequence':seq, 'expires_at':issued+.35}

    def tick(self):
        with self.lock:
            if self.compute_token and self.monotonic() >= self.compute_deadline:
                self.compute_token = None
                self.compute_deadline = 0.
                self.error = 'Compute reservation expired; motion remains stopped'
            if self.deadline and self.monotonic() >= self.deadline:
                self.error = 'Driving stopped because operator input expired; release and press a control again'
                try:
                    self._stop()
                except Exception as exc:
                    self.error = str(exc)

    def _watch(self):
        while not self.closed.wait(.025):
            self.tick()

    def state(self):
        with self.lock:
            return dict(run_id=self.run_id, episode_id=self.episode_id, mode=self.mode,
                        control_token=self.token, sequence=self.sequence, server_time=self.clock(),
                        error=self.error, driving=self.deadline > self.monotonic(),
                        compute_active=bool(self.compute_token),
                        skill_active=self.executor.processing_action_sequence)

    def close(self):
        self.closed.set()
        if self.thread is not None:
            self.thread.join(timeout=1)
        with self.lock:
            self._stop()
