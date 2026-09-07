const $=id=>document.getElementById(id);
let jet=null,ws=null,stream=null,testMode=false,videoReady=false,sendTimer=null,frameSent=0,designateArmed=false,connecting=false;

document.querySelectorAll('.jets button').forEach(b=>b.onclick=()=>{
  jet=b.dataset.jet;
  document.querySelectorAll('.jets button').forEach(x=>x.classList.remove('sel'));
  b.classList.add('sel');
  $('jetBadge').textContent=jet;
  $('setupStatus').textContent=jet+' SELECTED';
});

$('connect').onclick=async()=>{
  if(!jet)return toast('SELECT A JET');
  if(connecting)return;
  connecting=true;
  $('setupStatus').textContent='REQUESTING CAMERA…';
  const ok=await camera();
  if(!ok){ connecting=false; return; }
  connect();
  connecting=false;
};

$('test').onclick=()=>{
  if(!jet)jet='JET-01';
  testMode=true;
  $('jetBadge').textContent=jet;
  $('video').style.display='none';
  $('testCanvas').style.display='block';
  testView();
  connect();
};

$('fire').onclick=()=>{if(ws?.readyState===1)ws.send(JSON.stringify({type:'fire'}));};
$('reset').onclick=()=>{if(ws?.readyState===1)ws.send(JSON.stringify({type:'reset'}));showBattle();};

// Tap/click the desired jet in the live view. The server remembers that jet's
// shape/appearance and can reacquire it after occlusion or an ID switch.
$('video').addEventListener('click',e=>designate(e));
$('testCanvas').addEventListener('click',e=>designate(e));
function designate(e){
  if(!ws||ws.readyState!==1)return;
  const r=e.currentTarget.getBoundingClientRect();
  const x=(e.clientX-r.left)/r.width, y=(e.clientY-r.top)/r.height;
  ws.send(JSON.stringify({type:'designate',x,y}));
  toast('TARGET DESIGNATED');
}

async function camera(){
  if(!window.isSecureContext){
    $('setupStatus').textContent='CAMERA NEEDS HTTPS';
    toast('OPEN THE HTTPS PHONE URL');
    return false;
  }
  if(!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia){
    $('setupStatus').textContent='CAMERA API UNAVAILABLE';
    toast('BROWSER CAMERA API UNAVAILABLE');
    return false;
  }
  try{
    // Ask explicitly for the rear camera. Some Android browsers reject the
    // old ideal-only constraint when the page is opened from a local IP.
    stream=await navigator.mediaDevices.getUserMedia({
      video:{facingMode:{exact:'environment'},width:{ideal:640},height:{ideal:360}},
      audio:false
    });
    $('video').srcObject=stream;
    $('video').style.display='block';
    $('testCanvas').style.display='none';
    await new Promise((resolve,reject)=>{
      const v=$('video');
      if(v.readyState>=2) return resolve();
      v.onloadedmetadata=()=>resolve();
      setTimeout(()=>reject(new Error('camera video timeout')),5000);
    });
    await $('video').play();
    videoReady=true;
    const track=stream.getVideoTracks()[0];
    $('setupStatus').textContent='CAMERA READY // '+(track?.label||'VIDEO');
    return true;
  }catch(e){
    console.error('getUserMedia failed:',e.name,e.message);
    videoReady=false;
    if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}
    $('setupStatus').textContent='CAMERA ERROR: '+(e.name||'UNKNOWN');
    toast(e.name==='NotAllowedError'?'CAMERA PERMISSION DENIED':
          e.name==='NotFoundError'?'NO CAMERA FOUND':
          e.name==='OverconstrainedError'?'REAR CAMERA UNAVAILABLE':'CAMERA FAILED');
    return false;
  }
}

function connect(){
  if(ws&&ws.readyState===WebSocket.OPEN)return;
  const p=location.protocol==='https:'?'wss:':'ws:';
  $('setupStatus').textContent='CONNECTING COMMAND LINK…';
  ws=new WebSocket(p+'//'+location.host+'/ws');
  ws.onopen=()=>{
    $('ws').textContent='ONLINE';
    $('setupStatus').textContent='CONNECTED — STREAMING';
    ws.send(JSON.stringify({type:'hello',jet}));
    showBattle();
    startSendingFrames();
  };
  ws.onerror=(e)=>{
    console.error('WebSocket error',e);
    $('ws').textContent='ERROR';
    $('setupStatus').textContent='WEBSOCKET ERROR';
    toast('COMMAND LINK FAILED');
  };
  ws.onclose=()=>{
    $('ws').textContent='OFF';
    stopSendingFrames();
    toast('CONNECTION LOST');
  };
  ws.onmessage=e=>{
    let m;try{m=JSON.parse(e.data)}catch{return}
    if(m.type==='state')render(m);
    if(m.type==='event')event(m);
    if(m.type==='welcome')toast(m.jet+' LINKED');
  };
}

function startSendingFrames(){
  if(sendTimer)return;
  const c=document.createElement('canvas'),x=c.getContext('2d');
  c.width=640;c.height=360;
  sendTimer=setInterval(()=>{
    if(!ws||ws.readyState!==WebSocket.OPEN)return;
    if(testMode){
      x.fillStyle='#071016';x.fillRect(0,0,640,360);
      x.fillStyle='#d9f7ff';
      const t=Date.now()/600,px=320+Math.sin(t)*170,py=180+Math.cos(t*1.3)*75;
      // Simple fighter silhouette for deterministic end-to-end testing.
      x.beginPath();x.moveTo(px+32,py);x.lineTo(px-8,py-8);x.lineTo(px-30,py-24);x.lineTo(px-21,py-5);x.lineTo(px-42,py-1);x.lineTo(px-21,py+5);x.lineTo(px-30,py+24);x.lineTo(px-8,py+8);x.closePath();x.fill();
    }else if(videoReady){
      x.drawImage($('video'),0,0,640,360);
    }else return;
    frameSent++;
    $('frame').textContent=frameSent;
    ws.send(JSON.stringify({type:'frame',data:c.toDataURL('image/jpeg',.55)}));
  },180);
}
function stopSendingFrames(){if(sendTimer){clearInterval(sendTimer);sendTimer=null}}

// Android can suspend camera playback when the tab briefly loses focus.
// Resume the track/video when the pilot returns to the page.
document.addEventListener('visibilitychange', async()=>{
  if(document.visibilityState==='visible' && stream){
    try{ stream.getVideoTracks().forEach(t=>t.enabled=true); await $('video').play(); videoReady=true; }catch(e){ console.warn('camera resume failed',e); }
  }
});

function render(m){
  $('round').textContent=m.game.round;
  $('gameStatus').textContent=m.game.status;
  $('hp').textContent=m.game.health[jet]?'●':'○';
  let s=m.states[jet]||{},p=Number(s.lock_progress||0),locked=!!s.locked;
  $('track').textContent=s.state||'SEARCHING';
  $('lock').textContent=locked?'LOCKED':Math.round(p*100)+'%';
  $('targetState').textContent=locked?'TARGET LOCK':s.target?'TRACKING':'NO TARGET';
  $('lockText').textContent=locked?'LOCK':s.state||'SEARCHING';
  $('progress').style.width=(p*100)+'%';
  $('prediction').textContent=s.prediction?`PREDICTION: ${s.prediction[0].toFixed(2)}, ${s.prediction[1].toFixed(2)}`:'PREDICTION: --';
  $('identity').textContent=s.remembered?'REMEMBERED':'NEW TARGET';
  $('fire').disabled=!locked||m.game.status==='DEAD';
  if(s.state==='OCCLUDED' && (s.occlusion_age||0)>=0.45)toast('OCCLUSION // PREDICTING');
}

function event(m){
  if(m.event==='hit'){
    showDeath(m.attacker);
    if(navigator.vibrate)navigator.vibrate([220,100,220,100,550]);
    beep(true);
  }else if(m.event==='confirmed_hit'){
    toast('HIT CONFIRMED');beep(false);
  }else if(m.event==='miss')toast('MISS // NO LOCK');
  else if(m.event==='cooldown')toast('COOLDOWN '+m.remaining.toFixed(1)+'s');
  else if(m.event==='designated')toast('TARGET DESIGNATED');
}
function showBattle(){$('setup').classList.add('hidden');$('battle').classList.remove('hidden');$('death').classList.add('hidden');}
function showDeath(a){$('battle').classList.add('hidden');$('death').classList.remove('hidden');$('deathBy').textContent=a+' // HIT CONFIRMED';}
function toast(t){let e=$('toast');e.textContent=t;e.style.display='block';clearTimeout(window.tt);window.tt=setTimeout(()=>e.style.display='none',1400)}
function beep(long){try{let C=window.AudioContext||window.webkitAudioContext;if(!C)return;let c=new C,o=c.createOscillator(),g=c.createGain();o.connect(g);g.connect(c.destination);o.frequency.value=long?90:520;g.gain.value=.12;o.start();o.stop(c.currentTime+(long?.7:.12))}catch(e){}}
function testView(){
  let c=$('testCanvas'),x=c.getContext('2d');
  function d(){if(!testMode)return;c.width=innerWidth*devicePixelRatio;c.height=innerHeight*devicePixelRatio;x.save();x.scale(devicePixelRatio,devicePixelRatio);x.fillStyle='#071016';x.fillRect(0,0,innerWidth,innerHeight);x.fillStyle='#d9f7ff';let t=Date.now()/700,px=innerWidth/2+Math.sin(t)*innerWidth*.25,py=innerHeight/2+Math.cos(t*1.3)*innerHeight*.18;x.beginPath();x.moveTo(px+40,py);x.lineTo(px-10,py-10);x.lineTo(px-35,py-30);x.lineTo(px-24,py-6);x.lineTo(px-50,py);x.lineTo(px-24,py+6);x.lineTo(px-35,py+30);x.lineTo(px-10,py+10);x.closePath();x.fill();x.restore();requestAnimationFrame(d)}d();
}
