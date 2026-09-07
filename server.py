import argparse, asyncio, base64, json, socket, ssl, subprocess, time
from pathlib import Path
import cv2, numpy as np
from aiohttp import web, WSMsgType
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
from ai.target_manager import TargetManager

ROOT=Path(__file__).resolve().parent
CLIENTS={}
FRAMES={"JET-01":None,"JET-02":None}
STATES={"JET-01":{},"JET-02":{}}
MANAGERS={"JET-01":TargetManager(),"JET-02":TargetManager()}
MODEL=None
CONF=.20
LOCK_TIME=.85
TARGET_ZONE=.22
FIRE_COOLDOWN=1.5
GAME={"round":1,"status":"WAITING","winner":None,"last_hit":None,
      "health":{"JET-01":1,"JET-02":1},"fire_cooldown":{"JET-01":0.0,"JET-02":0.0}}
frame_counter=0

def decode_frame(data):
    if data.startswith("data:image"):
        data=data.split(",",1)[1]
    try:
        arr=np.frombuffer(base64.b64decode(data),dtype=np.uint8)
        return cv2.imdecode(arr,cv2.IMREAD_COLOR)
    except Exception:
        return None

def ai_process(jet, frame):
    m=MANAGERS[jet]
    if MODEL is None:
        m.update([],frame.shape[1],frame.shape[0]); return m.snapshot()
    try:
        r=MODEL.track(source=frame,persist=True,tracker="bytetrack.yaml",
                      conf=CONF,verbose=False)[0]
        ds=[]
        if r.boxes is not None:
            for b in r.boxes:
                xy=b.xyxy[0].cpu().numpy().tolist()
                c=float(b.conf[0].cpu().item())
                tid=int(b.id[0].cpu().item()) if b.id is not None else -1
                ds.append({"xyxy":xy,"conf":c,"id":tid})
        m.update(ds,frame.shape[1],frame.shape[0])
    except Exception as e:
        m.last_error=str(e)
    return m.snapshot()

def compute_lock(jet,state):
    now=time.time(); s=STATES[jet]
    t=state.get("target")
    inside=False
    if t:
        inside=abs(t["cx"]-.5)<=TARGET_ZONE and abs(t["cy"]-.5)<=TARGET_ZONE
    if inside:
        s.setdefault("lock_started",0)
        if not s["lock_started"]: s["lock_started"]=now
    else:
        s["lock_started"]=0
    p=(now-s["lock_started"])/LOCK_TIME if s.get("lock_started") else 0
    return p>=1.0,min(1.0,p)

async def send_to(cid,msg):
    ws=CLIENTS.get(cid)
    if ws:
        try: await ws.send_str(json.dumps(msg))
        except Exception: pass

async def broadcast(msg):
    payload=json.dumps(msg)
    for cid,ws in list(CLIENTS.items()):
        try: await ws.send_str(payload)
        except Exception: pass

def reset_round():
    GAME.update({"round":GAME["round"]+1,"status":"WAITING","winner":None,"last_hit":None,
                 "health":{"JET-01":1,"JET-02":1},"fire_cooldown":{"JET-01":0.0,"JET-02":0.0}})
    for s in STATES.values():
        s["lock_started"]=0

async def send_state():
    await broadcast({"type":"state","game":GAME,"states":STATES})

async def fire(attacker):
    target="JET-02" if attacker=="JET-01" else "JET-01"
    now=time.time()
    if GAME["status"]=="DEAD": return
    if now<GAME["fire_cooldown"][attacker]:
        await send_to(attacker,{"type":"event","event":"cooldown",
                                "remaining":GAME["fire_cooldown"][attacker]-now}); return
    GAME["fire_cooldown"][attacker]=now+FIRE_COOLDOWN
    GAME["status"]="ACTIVE"
    if not STATES[attacker].get("locked",False):
        await send_to(attacker,{"type":"event","event":"miss","reason":"NO_LOCK"})
        await send_state(); return
    GAME["health"][target]=0; GAME["status"]="DEAD"; GAME["winner"]=attacker
    GAME["last_hit"]={"attacker":attacker,"target":target,"time":now}
    await send_to(target,{"type":"event","event":"hit","attacker":attacker,"target":target})
    await send_to(attacker,{"type":"event","event":"confirmed_hit","target":target})
    await send_state()
    await asyncio.sleep(4)
    reset_round(); await send_state()

async def ai_loop():
    global frame_counter
    while True:
        await asyncio.sleep(.08)
        for jet,frame in list(FRAMES.items()):
            if frame is None: continue
            frame_counter+=1
            h,w=frame.shape[:2]
            if w>640:
                frame=cv2.resize(frame,(640,int(h*640/w)))
            state=ai_process(jet,frame)
            locked,progress=compute_lock(jet,state)
            state["locked"]=locked; state["lock_progress"]=progress
            STATES[jet]=state
        if frame_counter%2==0:
            await send_state()

async def index(_): raise web.HTTPFound("/phone")
async def file(request, folder):
    p=folder/request.match_info["path"]
    if not p.exists(): raise web.HTTPNotFound()
    return web.FileResponse(p)

async def ws_handler(request):
    ws=web.WebSocketResponse(max_msg_size=4*1024*1024); await ws.prepare(request)
    cid=None
    try:
        async for msg in ws:
            if msg.type!=WSMsgType.TEXT: continue
            try: d=json.loads(msg.data)
            except: continue
            typ=d.get("type")
            if typ=="hello":
                cid=d.get("jet")
                if cid in ("JET-01","JET-02","DASHBOARD"):
                    CLIENTS[cid]=ws
                    await send_to(cid,{"type":"welcome","jet":cid})
                    await send_state()
            elif typ=="frame" and cid in FRAMES:
                f=decode_frame(d.get("data",""))
                if f is not None: FRAMES[cid]=f
            elif typ=="fire" and cid in FRAMES: await fire(cid)
            elif typ=="reset" and cid in ("JET-01","JET-02","DASHBOARD"):
                reset_round(); await send_state()
    finally:
        if cid and CLIENTS.get(cid) is ws: CLIENTS.pop(cid,None)
        if cid in FRAMES: FRAMES[cid]=None
    return ws

def local_ip():
    s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8",80)); return s.getsockname()[0]
    except: return "127.0.0.1"
    finally: s.close()

def ssl_context(ip=None):
    """Create a locally trusted CA + an IP-address certificate.

    The old certificate was CN-only, so browsers complained that the cert did
    not match 172.20.10.2.  This version signs the server cert with a local CA
    and includes the current LAN IP in subjectAltName.

    To remove the browser warning completely, install trinetra-ca.pem as a
    trusted CA on the phone/Mac.  The server cert itself is then trusted for
    https://<LAN-IP>:8443.
    """
    ip = ip or local_ip()
    ca_cert = ROOT / "trinetra-ca.pem"
    ca_key = ROOT / "trinetra-ca-key.pem"
    cert = ROOT / "cert.pem"
    key = ROOT / "key.pem"
    cfg = ROOT / ".trinetra-openssl.cnf"

    try:
        # One local CA persists across server restarts.
        if not ca_cert.exists() or not ca_key.exists():
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                "-keyout", str(ca_key), "-out", str(ca_cert), "-days", "3650",
                "-subj", "/CN=Trinetra Local CA"
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Include the LAN IP in SAN. This fixes the "certificate does not match
        # the URL" error when users browse to https://172.x.x.x:8443.
        cfg.write_text(
            "[req]\n"
            "distinguished_name = req_distinguished_name\n"
            "req_extensions = req_ext\n"
            "prompt = no\n"
            "[req_distinguished_name]\n"
            "CN = Trinetra Server\n"
            "[req_ext]\n"
            "subjectAltName = @alt_names\n"
            "[alt_names]\n"
            f"IP.1 = {ip}\n"
            "DNS.1 = localhost\n"
        )

        csr = ROOT / ".trinetra-server.csr"
        subprocess.run([
            "openssl", "req", "-new", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(key), "-out", str(csr), "-config", str(cfg)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        subprocess.run([
            "openssl", "x509", "-req", "-in", str(csr),
            "-CA", str(ca_cert), "-CAkey", str(ca_key),
            "-CAcreateserial", "-out", str(cert), "-days", "825",
            "-sha256", "-extfile", str(cfg), "-extensions", "req_ext"
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ctx.load_cert_chain(str(cert), str(key))
        return ctx
    except Exception as e:
        print("HTTPS certificate setup failed:", e, flush=True)
        return None
    finally:
        for tmp in (cfg, ROOT / ".trinetra-server.csr", ROOT / "cert.pem.srl"):
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass

async def async_main():
    global MODEL, CONF
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=.20)
    ap.add_argument("--http", action="store_true", help="deprecated: HTTP on 8080 is now always enabled")
    a=ap.parse_args(); CONF=a.conf

    if YOLO:
        try:
            print("Loading", a.model, flush=True)
            MODEL=YOLO(a.model)
            print("YOLO ready", flush=True)
        except Exception as e:
            print("WARNING: YOLO unavailable:", e, flush=True)
    else:
        print("WARNING: ultralytics import failed; install requirements.txt", flush=True)

    app=web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/health", lambda r: web.json_response({"ok": True, "service": "TRINETRA V2"}))
    app.router.add_get("/phone", lambda r: web.FileResponse(ROOT/"phone/index.html"))
    app.router.add_get("/dashboard", lambda r: web.FileResponse(ROOT/"dashboard/index.html"))
    app.router.add_get("/phone/{path:.*}", lambda r: file(r, ROOT/"phone"))
    app.router.add_get("/dashboard/{path:.*}", lambda r: file(r, ROOT/"dashboard"))
    app.router.add_get("/ws", ws_handler)

    runner=web.AppRunner(app)
    await runner.setup()
    tasks=[]
    try:
        tasks.append(asyncio.create_task(ai_loop()))

        ip=local_ip()
        # HTTPS is the primary phone endpoint because camera access requires a secure context.
        # The certificate contains the LAN IP in subjectAltName.
        ctx=ssl_context(ip)
        if ctx is None:
            raise RuntimeError("Could not create/load cert.pem and key.pem. Make sure openssl is installed.")

        https_site=web.TCPSite(runner, "0.0.0.0", 8443, ssl_context=ctx)
        await https_site.start()

        # Also serve HTTP by default. The Mac dashboard can use this URL without
        # any certificate warning; phones must use HTTPS for camera access.
        http_site=web.TCPSite(runner, "0.0.0.0", 8080)
        await http_site.start()

        print("", flush=True)
        print("="*60, flush=True)
        print("TRINETRA V2 SERVER IS LISTENING", flush=True)
        print(f"HTTPS Phone:  https://{ip}:8443/phone", flush=True)
        print(f"HTTPS Dash:   https://{ip}:8443/dashboard", flush=True)
        print(f"HTTP Dash:    http://{ip}:8080/dashboard", flush=True)
        print("Health:       https://127.0.0.1:8443/health", flush=True)
        print(f"CA file:      {ROOT / 'trinetra-ca.pem'}", flush=True)
        print("="*60, flush=True)
        print("Open the HTTPS phone URL on both phones.", flush=True)

        await asyncio.Event().wait()
    finally:
        for t in tasks:
            t.cancel()
        await runner.cleanup()


def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nTRINETRA server stopped.", flush=True)


if __name__ == "__main__":
    main()
