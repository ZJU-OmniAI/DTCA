'use strict';
(() => {
  const $ = (selector, parent = document) => parent.querySelector(selector);
  const $$ = (selector, parent = document) => Array.from(parent.querySelectorAll(selector));
  const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

  const menu = $('.menu-button');
  menu.addEventListener('click', () => {
    const open = menu.getAttribute('aria-expanded') !== 'true';
    menu.setAttribute('aria-expanded', String(open));
    menu.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
    $('#navigation').classList.toggle('open', open);
  });
  $$('#navigation a').forEach(link => link.addEventListener('click', () => {
    menu.setAttribute('aria-expanded', 'false');
    menu.setAttribute('aria-label', 'Open navigation');
    $('#navigation').classList.remove('open');
  }));
  const navigation = $$('#navigation a');
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        navigation.forEach(link => link.classList.toggle('active', link.hash === `#${entry.target.id}`));
      });
    }, {rootMargin:'-15% 0px -65% 0px'});
    $$('main section[id]').forEach(section => observer.observe(section));
  }
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const activeMotions = new Set();
  const elementMotion = new WeakMap();
  function move(element, keyframes, options = {}) {
    if (!element || reducedMotion.matches || typeof element.animate !== 'function') return;
    elementMotion.get(element)?.cancel();
    const animation = element.animate(keyframes, {
      duration: 360, easing: 'cubic-bezier(.2,.75,.25,1)', ...options,
    });
    elementMotion.set(element, animation); activeMotions.add(animation);
    const finish = () => {
      activeMotions.delete(animation);
      if (elementMotion.get(element) === animation) elementMotion.delete(element);
    };
    animation.finished.then(finish, finish);
  }
  function pulse(element, delay = 0) {
    move(element, [
      {opacity:.5, transform:'translateY(3px) scale(.97)'},
      {opacity:1, transform:'translateY(0) scale(1.035)', offset:.6},
      {opacity:1, transform:'translateY(0) scale(1)'},
    ], {duration:400, delay});
  }
  reducedMotion.addEventListener('change', () => {
    if (reducedMotion.matches) [...activeMotions].forEach(animation => animation.cancel());
  });

  // One illustrative selected state supplies both credit and entropy views.
  // Fixed POS weights come from offline contribution estimates.
  const targets = [
    {id:'T₁', token:'and', pos:'CCONJ', weight:window.DTCA_OFFLINE_POS.weights.CCONJ, flow:-.18},
    {id:'T₂', token:'not', pos:'PART', weight:window.DTCA_OFFLINE_POS.weights.PART, flow:.12},
    {id:'T₃', token:'quit', pos:'VERB', weight:window.DTCA_OFFLINE_POS.weights.VERB, flow:.18},
  ];
  const baseAdvantage = 1;
  const meanWeight = targets.reduce((sum,t) => sum+t.weight,0)/targets.length;
  const exampleAdvantages = targets.map(t => baseAdvantage*t.weight/meanWeight);
  const exampleFlows = targets.map(t => t.flow);
  const positiveMass = exampleFlows.reduce((sum,v) => sum+Math.max(v,0),0);
  const negativeMass = exampleFlows.reduce((sum,v) => sum+Math.max(-v,0),0);
  const flowLambda = Math.max(-.95,Math.min(.95,(negativeMass-positiveMass)/(negativeMass+positiveMass)));
  const flowMultipliers = exampleFlows.map(v => v>0 ? 1+flowLambda : 1-flowLambda);
  const rescaledAdvantages = exampleAdvantages.map((v,i) => v*flowMultipliers[i]);
  const totalCredit = exampleAdvantages.reduce((a,b)=>a+b,0);
  const advantageNormalization = totalCredit/rescaledAdvantages.reduce((a,b)=>a+b,0);
  const calibratedAdvantages = rescaledAdvantages.map(v=>v*advantageNormalization);
  const calibratedFlows = exampleFlows.map((v,i)=>v*flowMultipliers[i]*advantageNormalization);
  // Scale signed bars to fit the entropy chart.
  const entropyBarScale = 42;
  const signedFlow = value => Math.abs(value)<1e-10 ? '0.000' : `${value>0?'+':'−'}${Math.abs(value).toFixed(3)}`;
  let assigned = true;
  let calibrated = true;
  let contributionCommitted = true;
  // Isolate the reveal of "and" in this example.
  const sequence = [
    {visible:'We'}, {visible:'should'}, {visible:'try'},
    {visible:targets[0].token, target:0, committed:true}, {visible:'[MASK]', target:1, committed:false}, {visible:'[MASK]', target:2, committed:false},
  ];
  const contributionConfidences = [.39,.28,.23];
  const isolatedConfidences = [.42,.44];
  // Sum maximum vocabulary probabilities over the same remaining masks.
  const remainingMaskCount = contributionConfidences.length-1;
  const confidenceBefore = contributionConfidences.slice(1).reduce((a,b)=>a+b,0);
  const confidenceAfter = isolatedConfidences.reduce((a,b)=>a+b,0);
  const contributionGain = Math.max(0,confidenceAfter-confidenceBefore);
  function tokenSequence(kind) {
    const entropy = kind === 'entropy';
    const contribution = kind === 'contribution';
    return sequence.map((position,slot) => {
      const t = targets[position.target];
      const visible = contribution && t ? '[MASK]' : position.visible;
      const isCommitted = position.committed && !contribution;
      // Label unresolved positions with the final token and its POS.
      const posLabel = t && !position.committed ? `<strong>${escape(t.token)}</strong> · ${escape(t.pos)}` : t ? escape(t.pos) : '';
      const metadata = t ? `<small class="mask-id" title="Final token: ${escape(t.token)}; POS: ${escape(t.pos)}" aria-label="Final token: ${escape(t.token)}; POS: ${escape(t.pos)}">${posLabel}</small>${kind === 'credit' ? `<small class="mask-id">w = ${t.weight.toFixed(3)}</small>` : ''}` : '';
      const token = `<div class="noise-token ${t ? isCommitted?'committed':'masked' : ''}"><b class="token-value">${escape(visible)}</b>${metadata}</div>`;
      if (!t) return `<div class="token-lane context-lane" data-slot="${slot}">${token}</div>`;
      if (contribution) {
        const before = contributionConfidences[position.target];
        const confidence = position.target === 0
          ? ''
          : `<div class="mask-confidence" data-target="${position.target}"><div class="confidence-transition"><span class="mask-before">${before.toFixed(3)}</span><span class="change-arrow" aria-hidden="true">→</span><span class="mask-after">${isolatedConfidences[position.target-1].toFixed(3)}</span></div><strong class="mask-gain">${signedFlow(isolatedConfidences[position.target-1]-before)}</strong></div>`;
        return `<div class="token-lane context-lane contribution-target" data-slot="${slot}">${token}${confidence}</div>`;
      }
      const value = entropy ? t.flow : baseAdvantage;
      const height = entropy ? Math.abs(value)/.30*entropyBarScale : value/1.5*180;
      const direction = value<0 ? 'negative' : 'positive';
      const cellClass = entropy ? 'flow-cell adjusted-flow' : 'credit-cell';
      return `<div class="token-lane target-lane" data-slot="${slot}" data-target="${position.target}">${token}<div class="position-chart ${entropy?'entropy-chart':'advantage-chart'}"><div class="metric-series before-series"><i class="metric-bar ${direction}" style="height:${height}px"><span>${entropy?signedFlow(value):value.toFixed(3)}</span></i></div><div class="metric-series after-series ${cellClass}" data-target="${position.target}"><i class="metric-bar ${direction}" style="height:0"><span>—</span></i></div></div></div>`;
    }).join('');
  }
  $('#contribution-sequence').innerHTML = tokenSequence('contribution');
  $('#credit-sequence').innerHTML = tokenSequence('credit');
  $('#entropy-sequence').innerHTML = tokenSequence('entropy');
  function highlightTargets(panel) {
    $$('.target-lane .noise-token',panel).forEach((token,index) => pulse(token,index*55));
  }
  function updateContribution(animate = true) {
    const token = $('#contribution-sequence [data-slot="3"] .noise-token');
    $('.token-value',token).textContent = contributionCommitted ? targets[0].token : '[MASK]';
    token.classList.toggle('committed',contributionCommitted);
    token.classList.toggle('masked',!contributionCommitted);
    $('#contribution-after').textContent = contributionCommitted ? confidenceAfter.toFixed(3) : '—';
    $('#contribution-after-bar').style.width = `${contributionCommitted ? confidenceAfter/remainingMaskCount*100 : 0}%`;
    $('#contribution-gain').textContent = contributionCommitted ? signedFlow(contributionGain) : '—';
    if (animate) {pulse(token); pulse($('#contribution-gain'),120);}
  }
  function updateCredit(animate = true) {
    const advantages = assigned ? exampleAdvantages : targets.map(() => baseAdvantage);
    $$('.credit-cell').forEach((cell,index) => {
      $('.metric-bar span',cell).textContent = assigned ? advantages[index].toFixed(3) : '—';
      $('.metric-bar',cell).style.height = `${assigned ? advantages[index]/1.5*180 : 0}px`;
      if (animate) pulse(cell,index*60);
    });
    $('.allocation-layout').classList.toggle('is-assigned',assigned);
    if (animate) highlightTargets($('#panel-credit'));
  }
  function updateCalibration(animate = true) {
    const currentFlows = calibrated ? calibratedFlows : exampleFlows;
    $$('.adjusted-flow').forEach((cell,index) => {
      const value = calibrated ? currentFlows[index] : 0;
      $('.metric-bar span',cell).textContent = calibrated ? signedFlow(value) : '—';
      $('.metric-bar',cell).style.height = `${Math.abs(value)/.30*entropyBarScale}px`;
      if (animate) pulse(cell,index*70);
    });
    $('#net-flow-before').textContent = signedFlow(exampleFlows.reduce((a,b)=>a+b,0));
    $('#net-flow').textContent = signedFlow(currentFlows.reduce((a,b)=>a+b,0));
    $('.flow-result').classList.toggle('is-balanced',calibrated);
    $('#panel-calibration .calibration-layout').classList.toggle('is-calibrated',calibrated);
    if (animate) {
      highlightTargets($('#panel-calibration')); pulse($('#net-flow'),160);
    }
  }

  const scores = [.03,.06,.14,.27,.31,.19];
  function updateSelection(animate = true) {
    const target = Number($('#mass-slider').value)/100;
    const order = scores.map((score,index) => ({score,index})).sort((a,b) => b.score-a.score);
    const selected = new Set();
    let coverage = 0;
    for (const state of order) {
      if (coverage+1e-10 >= target && selected.size >= 2) break;
      selected.add(state.index); coverage += state.score;
    }
    $('#mass-value').textContent = `${Math.round(target*100)}%`;
    $('#selection-result').textContent = `${selected.size} states · ${Math.round(coverage*100)}% covered`;
    if (!$('#state-bars').children.length) {
      $('#state-bars').innerHTML = scores.map((score,index) => `<div class="state-column${index===4?' featured-state':''}" style="height:${score/.31*160}px" role="img"><strong>${score.toFixed(2)}</strong><span>State<br>${index+1}</span></div>`).join('');
      animate = false;
    }
    $$('.state-column').forEach((column,index) => {
      const active = selected.has(index);
      const changed = active !== column.classList.contains('selected');
      column.classList.toggle('selected',active);
      column.dataset.marker = active ? '✓' : '';
      column.setAttribute('aria-label',`State ${index+1}, priority ${scores[index].toFixed(2)}, ${active?'selected':'not selected'}${index===4?', shared example':''}`);
      if (animate && changed) move(column,active
        ? [{transform:'translateY(0) scaleY(.96)'},{transform:'translateY(-5px) scaleY(1.03)',offset:.5},{transform:'translateY(0) scaleY(1)'}]
        : [{opacity:.5},{opacity:1}],{duration:360});
    });
  }
  $('#mass-slider').addEventListener('input',() => updateSelection());
  // Animate comparisons once on entering the viewport.
  function animateStage(name, entrance = false) {
    const panel = $(`#panel-${name}`);
    if (entrance) move(panel,[{opacity:.5,transform:'translateY(16px)'},{opacity:1,transform:'translateY(0)'}],{duration:380});
    if (name === 'contribution') {
      contributionCommitted = true; updateContribution(false);
      const bar = $('#contribution-after-bar');
      move(bar,[{width:'0%'},{width:bar.style.width}],{duration:480});
      pulse($('#contribution-sequence [data-slot="3"] .noise-token'),80);
      pulse($('#contribution-gain'),120);
      $$('.mask-confidence',panel).forEach((cell,index) => {pulse($('.mask-after',cell),100+index*70); pulse($('.mask-gain',cell),160+index*70);});
    } else if (name === 'selection') {
      $$('.state-column',panel).forEach((column,index) => move(column,
        [{transform:'scaleY(.12)',opacity:.4},{transform:'scaleY(1)',opacity:1}],{duration:420,delay:index*35}));
    } else {
      if (name === 'credit') {assigned = true; updateCredit(false);}
      else {assigned = true; calibrated = true; updateCredit(false); updateCalibration(false);}
      $$('.after-series .metric-bar',panel).forEach((bar,index) => {
        move(bar,[{height:'0px'},{height:bar.style.height}],{duration:480,delay:index*60});
      });
      highlightTargets(panel);
      if (name === 'calibration') pulse($('#net-flow'),120);
    }
  }
  updateContribution(false); updateCredit(false); updateCalibration(false); updateSelection(false);
  if ('IntersectionObserver' in window) {
    const stageObserver = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        animateStage(entry.target.dataset.step,true);
        stageObserver.unobserve(entry.target);
      });
    },{threshold:.18});
    $$('.flow-step').forEach(panel => stageObserver.observe(panel));
  }

  let toastTimer;
  function toast(message) {
    clearTimeout(toastTimer); $('#toast').textContent = message; $('#toast').classList.add('visible');
    toastTimer = setTimeout(() => $('#toast').classList.remove('visible'), 2400);
  }
  async function copyText(text) {
    try {await navigator.clipboard.writeText(text); return true;} catch (_) {
      const textarea = document.createElement('textarea'); textarea.value = text;
      textarea.style.cssText = 'position:fixed;left:-10000px;top:0'; document.body.append(textarea); textarea.select();
      const copied = document.execCommand('copy'); textarea.remove(); return copied;
    }
  }
  $$('.copy-button').forEach(button => button.addEventListener('click', async () => {
    const label = button.dataset.originalLabel || button.textContent; button.dataset.originalLabel = label;
    const copied = await copyText($(`#${button.dataset.copy}`).textContent);
    toast(copied ? 'Copied to clipboard' : 'Select the text to copy it.');
    if (copied) {button.textContent = 'Copied ✓'; setTimeout(() => button.textContent = label, 2400);}
  }));
})();
