/* handwriting-recognizer — MNIST CNN (TensorFlow.js) trained live in the browser on a
   real MNIST subset, drawn on canvas with touch support. Top-3 predictions shown.
   HONEST: trained on MNIST digits; accuracy shown is the real test-set number. */
(function(){
'use strict';
const $=id=>document.getElementById(id);
const IMG=28,TRAIN_N=4000,TEST_N=1200,EPOCHS=2,BATCH=64;
let model=null,xTrain,yTrain,xTest,yTest,ready=false,drawing=false,strokes=0;
async function loadData(progress){
  const imgRes=await fetch('https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png');
  const imgBlob=await imgRes.blob();
  const img=await createImageBitmap(imgBlob);
  const c=document.createElement('canvas');c.width=img.width;c.height=img.height;
  const ctx=c.getContext('2d');ctx.drawImage(img,0,0);
  const labRes=await fetch('https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8');
  const labels=new Uint8Array(await labRes.arrayBuffer());
  const total=TRAIN_N+TEST_N;
  const xs=new Float32Array(total*IMG*IMG);
  for(let i=0;i<total;i++){
    const d=ctx.getImageData((i%65)*IMG,Math.floor(i/65)*IMG,IMG,IMG).data;
    for(let j=0;j<IMG*IMG;j++)xs[i*IMG*IMG+j]=d[j*4]/255;
    if(progress&&i%2000===0)progress(Math.round(i/total*40));
  }
  const lab=labels.slice(0,total);
  xTrain=tf.tensor4d(xs.slice(0,TRAIN_N*IMG*IMG),[TRAIN_N,IMG,IMG,1]);
  xTest=tf.tensor4d(xs.slice(TRAIN_N*IMG*IMG),[TEST_N,IMG,IMG,1]);
  yTrain=tf.oneHot(lab.slice(0,TRAIN_N),10);
  yTest=tf.oneHot(lab.slice(TRAIN_N),10);
}
function buildModel(){
  const m=tf.sequential();
  m.add(tf.layers.conv2d({inputShape:[IMG,IMG,1],filters:8,kernelSize:3,activation:'relu'}));
  m.add(tf.layers.maxPooling2d({poolSize:2}));
  m.add(tf.layers.conv2d({filters:16,kernelSize:3,activation:'relu'}));
  m.add(tf.layers.maxPooling2d({poolSize:2}));
  m.add(tf.layers.flatten());
  m.add(tf.layers.dropout({rate:0.25}));
  m.add(tf.layers.dense({units:64,activation:'relu'}));
  m.add(tf.layers.dense({units:10,activation:'softmax'}));
  m.compile({optimizer:'adam',loss:'categoricalCrossentropy',metrics:['accuracy']});
  return m;
}
async function train(){
  $('prog-wrap').classList.remove('hidden');
  try{await tf.setBackend('cpu');await tf.ready()}catch(e){console.warn('backend',e)}
  const prog=(p,msg)=>{$('prog-bar').style.width=Math.round(p)+'%';$('prog-msg').textContent=msg};
  prog(2,'Downloading MNIST (free Google-hosted dataset)…');
  await loadData(p=>prog(p,'Loading MNIST images… '+p+'%'));
  prog(45,'Building the network…');
  model=buildModel();
  let testAcc=null;
  await model.fit(xTrain,yTrain,{epochs:EPOCHS,batchSize:BATCH,validationData:[xTest,yTest],callbacks:{onEpochEnd:(e,logs)=>{
    prog(45+(e+1)/EPOCHS*50,`Epoch ${e+1}/${EPOCHS} — loss ${logs.loss.toFixed(3)}, val acc ${(logs.val_acc*100).toFixed(1)}%`);
    testAcc=logs.val_acc;
  }}});
  prog(100,`Done — test accuracy ${(testAcc*100).toFixed(1)}% on ${TEST_N} unseen digits.`);
  $('acc-pill').innerHTML=`<span class="pill ok">Test accuracy ${(testAcc*100).toFixed(1)}%</span> <span class="hint">measured on ${TEST_N} held-out MNIST digits · ${TRAIN_N} train · ${EPOCHS} epochs</span>`;
  ready=true;
  $('pad-hint').textContent='Draw a digit (0-9) with your mouse or finger.';
}
function preprocess(){
  const src=$('pad');
  const c=document.createElement('canvas');c.width=IMG;c.height=IMG;
  const ctx=c.getContext('2d');
  ctx.fillStyle='#000';ctx.fillRect(0,0,IMG,IMG);
  ctx.drawImage(src,0,0,IMG,IMG);
  let t=tf.browser.fromPixels(c,1).toFloat().div(255).reshape([1,IMG,IMG,1]);
  return t;
}
async function predict(){
  if(!ready){pkToast('Model is still training — hang on');return}
  if(strokes===0){pkToast('Draw a digit first');return}
  const t=preprocess();
  const out=model.predict(t);
  const probs=await out.data();t.dispose();out.dispose();
  const idx=[...probs.keys()].sort((a,b)=>probs[b]-probs[a]);
  const top=idx.slice(0,3);
  $('pred-digit').textContent=top[0];
  $('pred-conf').textContent=(probs[top[0]]*100).toFixed(1)+'% confident';
  $('pred-bars').innerHTML=top.map(i=>`<div class="pb-row"><span class="pb-d">${i}</span><div class="pb-track"><div class="pb-fill" style="width:${Math.max(2,probs[i]*100)}%"></div></div><span class="mono">${(probs[i]*100).toFixed(1)}%</span></div>`).join('');
  $('pred-card').classList.remove('hidden');
}
function clearPad(){
  const c=$('pad');const ctx=c.getContext('2d');
  ctx.fillStyle='#0a1220';ctx.fillRect(0,0,c.width,c.height);
  strokes=0;$('pred-card').classList.add('hidden');
}
function setupPad(){
  const c=$('pad');const ctx=c.getContext('2d');
  c.width=560;c.height=560;
  clearPad();
  ctx.lineWidth=34;ctx.lineCap='round';ctx.lineJoin='round';ctx.strokeStyle='#fff';
  let last=null;
  const pos=e=>{const r=c.getBoundingClientRect();const p=e.touches?e.touches[0]:e;return[(p.clientX-r.left)*c.width/r.width,(p.clientY-r.top)*c.height/r.height]};
  const start=e=>{e.preventDefault();drawing=true;strokes++;last=pos(e)};
  const move=e=>{if(!drawing)return;e.preventDefault();const p=pos(e);ctx.beginPath();ctx.moveTo(...last);ctx.lineTo(...p);ctx.stroke();last=p};
  const end=()=>{if(drawing){drawing=false;predict()}};
  c.addEventListener('mousedown',start);c.addEventListener('mousemove',move);addEventListener('mouseup',end);
  c.addEventListener('touchstart',start,{passive:false});c.addEventListener('touchmove',move,{passive:false});c.addEventListener('touchend',end);
}
document.addEventListener('DOMContentLoaded',()=>{
  setupPad();
  $('clear').onclick=clearPad;
  $('predict-btn').onclick=predict;
  train().catch(e=>{$('prog-msg').textContent='Training failed: '+e.message;console.error(e)});
});
})();
