module.exports=async(page)=>{
  // training takes a while: wait up to 4.5 min for ready state
  await page.waitForFunction(()=>document.querySelector('#acc-pill')&&/Test accuracy/.test(document.querySelector('#acc-pill').textContent),{timeout:270000});
  const acc=await page.$eval('#acc-pill',e=>e.textContent);
  // draw a "1": a vertical stroke
  const box=await (await page.$('#pad')).boundingBox();
  await page.mouse.move(box.x+box.width/2,box.y+box.height*0.2);
  await page.mouse.down();
  await page.mouse.move(box.x+box.width/2,box.y+box.height*0.8,{steps:12});
  await page.mouse.up();
  await page.waitForSelector('#pred-card:not(.hidden)',{timeout:15000});
  const digit=await page.$eval('#pred-digit',e=>e.textContent);
  const conf=await page.$eval('#pred-conf',e=>e.textContent);
  const bars=await page.$$eval('.pb-row',e=>e.length);
  return `trained (${acc.split('·')[0].trim()}), drew a stroke → predicted "${digit}" (${conf}), ${bars} bars`;
};
