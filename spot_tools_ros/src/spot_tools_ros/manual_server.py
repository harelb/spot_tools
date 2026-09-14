"""Optional authenticated operator endpoint hosted by the real executor node."""
import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from spot_executor.manual_control import ManualControl


class ManualServer:
    def __init__(self, node, port, token_file, run_id, episode_id):
        path=Path(token_file)
        if path.stat().st_mode & 0o077:
            raise ValueError('Manual endpoint token file must be private')
        token=path.read_text().strip()
        if len(token)<32 or not run_id or not episode_id:
            raise ValueError('Manual endpoint requires credentials and run/episode identities')
        self.node=node
        self.control=ManualControl(node.spot_executor,node.feedback_collector,
            run_id=run_id,episode_id=episode_id,ready=self.ready)
        endpoint=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):self.handle_request(False)
            def do_POST(self):self.handle_request(True)
            def handle_request(self,post):
                status=200
                try:
                    if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):
                        status=401;raise ValueError('Unauthorized')
                    if post:
                        length=int(self.headers.get('Content-Length',0))
                        if not 0<length<=4096:raise ValueError('Invalid request size')
                        request=json.loads(self.rfile.read(length))
                        operation={'/mode':endpoint.control.claim,'/teleop':endpoint.control.drive,
                            '/stop':endpoint.control.stop,'/compute':endpoint.control.reserve_compute,
                            '/compute_release':endpoint.control.release_compute}.get(self.path)
                        if operation is None:raise ValueError('Unsupported manual operation')
                        value=operation(request)
                    elif self.path=='/state':
                        value=endpoint.control.state()
                        value['ready']=True
                        value['motion_ready']=endpoint.ready()
                        value['reason']='Fresh collision coverage required' if not value['motion_ready'] else 'Spot Tools hardware motion available'
                    else:raise ValueError('Unknown endpoint')
                except Exception as exc:
                    if status==200:status=409
                    value={'detail':str(exc)}
                payload=json.dumps(value,allow_nan=False).encode()
                self.send_response(status);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        self.server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
        threading.Thread(target=self.server.serve_forever,daemon=True).start()

    def ready(self):
        try:
            interface=self.node.spot_interface
            if getattr(interface,'backend',None)=='isaac':
                health=interface.transport.call('health')
                return bool(health.get('motion_ready')) and not health.get('paused')
            # Shared endpoint remains opt-in; physical freshness comes from SDK.
            interface.get_state()
            manager=self.node.spot_executor.lease_manager
            return not manager.error and manager.owner_name.startswith('understanding')
        except Exception:
            return False

    def close(self):
        self.control.close()
        self.server.shutdown();self.server.server_close()
