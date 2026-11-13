/* Keep external references from replacing the document being read. */
(() => {
  for (const link of document.querySelectorAll('a[href]')) {
    if (link.hasAttribute('download')) continue;
    let destination;
    try { destination = new URL(link.href, document.baseURI); } catch { continue; }
    if (!['http:', 'https:'].includes(destination.protocol) || destination.origin === location.origin) continue;
    link.target = '_blank';
    link.relList.add('noopener', 'noreferrer');
  }
})();

/* Shared heading index: reads visible report sections, never experimental data. */
(() => {
  const main = document.querySelector('main');
  const index = document.querySelector('.page-index nav');
  const disclosure = document.querySelector('.page-index details');
  if (!main || !index || !disclosure) return;
  const mobile = matchMedia('(max-width: 800px)');
  function setDisclosure() { disclosure.open = !mobile.matches; }
  setDisclosure();
  mobile.addEventListener('change', setDisclosure);
  disclosure.addEventListener('toggle', () => {
    if (!mobile.matches && !disclosure.open) disclosure.open = true;
  });
  // IDs persist across dataset changes so existing section links stay usable.
  const used = new Set([...document.querySelectorAll('[id]')].map(el => el.id));
  main.querySelectorAll('h2, h3, h4').forEach((heading, i) => {
    if (!heading.id) {
      const base = heading.textContent.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'section';
      let id = base;
      if (used.has(id)) id += '-' + i;
      used.add(id);
      heading.id = id;
    }
  });
  let entries = [];
  // Keep explicit choices separate from unseen groups, which start expanded.
  const expanded = new Map();
  let scheduled = false;
  function selectCurrent() {
    scheduled = false;
    const threshold = document.querySelector('.site-header').getBoundingClientRect().bottom + 50;
    let current = entries[0];
    for (const entry of entries) {
      if (entry.target.getBoundingClientRect().top <= threshold) current = entry;
    }
    // A collapsed section represents whichever descendant is currently being read.
    for (let parent = current?.parent; parent; parent = parent.parent) {
      if (expanded.get(parent.target.id) === false) current = parent;
    }
    for (const entry of entries) {
      if (entry === current) entry.link.setAttribute('aria-current', 'location');
      else entry.link.removeAttribute('aria-current');
    }
  }
  function updateContextLinks() {
    const panel = main.querySelector('.dataset-panel:not([hidden])');
    const results = document.getElementById('results');
    const tab = panel && document.getElementById(panel.getAttribute('aria-labelledby'));
    if (results) results.textContent = tab ? 'Task: ' + tab.textContent : 'Results';
    for (const link of main.querySelectorAll('a[data-active-dataset-target]')) {
      const target = panel?.querySelector('[id="'+link.dataset.activeDatasetTarget+'-'+panel.id.replace('panel-', '')+'"]');
      if (target) link.href='#'+target.id;
    }
    for (const link of main.querySelectorAll('a[data-active-dataset-section]')) {
      const target = panel?.querySelector('[id^="dataset-'+link.dataset.activeDatasetSection+'-"]');
      if (target) link.href='#'+target.id;
    }
  }
  function setExpanded(entry, open) {
    expanded.set(entry.target.id, open);
    entry.childGroup.hidden = !open;
    entry.toggle.setAttribute('aria-expanded', String(open));
    entry.toggle.setAttribute('aria-label',
      (open ? 'Collapse' : 'Expand') + ' subheadings under ' + entry.link.textContent);
  }
  function revealIndex(target) {
    const entry = entries.find(item => item.target === target || item.target.contains(target));
    for (let parent = entry?.parent; parent; parent = parent.parent) setExpanded(parent, true);
    selectCurrent();
  }
  function rebuild() {
    updateContextLinks();
    const headings = [...main.querySelectorAll('h2, h3, h4')].filter(el => !el.closest('[hidden]'));
    const parents = [], roots = [];
    entries = headings.map(target => {
      const link = document.createElement('a');
      link.href = '#' + target.id;
      let label = target.textContent;
      if (target.matches('[id^="roc-title-"]')) label = 'Figure 1 · ROC';
      else if (target.matches('[id^="resource-title-"]')) label = 'Figure 2 · Resources';
      else if (target.matches('[id^="table-title-"]')) label = 'Table 1 · Results';
      link.textContent = label;
      if (target.tagName === 'H3') link.className = 'index-subsection';
      if (target.tagName === 'H4') link.className = 'index-detail';
      link.addEventListener('click', () => { if (mobile.matches) disclosure.open = false; });
      const level = Number(target.tagName.slice(1));
      while (parents.length && parents.at(-1).level >= level) parents.pop();
      const parent = parents.at(-1);
      const node = document.createElement('div');
      node.className = 'index-group';
      const row = document.createElement('div');
      row.className = 'index-row';
      row.append(link);
      node.append(row);
      const entry = { target, link, level, parent, node, row, children: [] };
      if (parent) parent.children.push(entry);
      else roots.push(entry);
      parents.push(entry);
      return entry;
    });
    for (const entry of entries) {
      if (!entry.children.length) continue;
      entry.toggle = document.createElement('button');
      entry.toggle.type = 'button';
      entry.toggle.className = 'index-toggle';
      const arrow = document.createElement('span');
      arrow.textContent = '▸';
      arrow.setAttribute('aria-hidden', 'true');
      entry.toggle.append(arrow);
      entry.childGroup = document.createElement('div');
      entry.childGroup.className = 'index-children';
      entry.childGroup.append(...entry.children.map(child => child.node));
      entry.row.prepend(entry.toggle);
      entry.node.append(entry.childGroup);
      setExpanded(entry, expanded.get(entry.target.id) ?? true);
      entry.toggle.addEventListener('click', () => {
        setExpanded(entry, !expanded.get(entry.target.id));
        selectCurrent();
      });
    }
    index.replaceChildren(...roots.map(entry => entry.node));
    selectCurrent();
  }
  function viewMessage(message) {
    let status=document.getElementById('report-view-message');
    if(!message) { status?.remove(); return; }
    if(!status) {
      status=document.createElement('p');status.id='report-view-message';
      status.className='selection-message';status.setAttribute('role','alert');
      main.prepend(status);
    }
    status.textContent=message;
    status.scrollIntoView();
  }
  // Reveal outer containers before following ordinary or preset section links.
  function revealTarget(target) {
    const ancestors=[];
    for(let ancestor=target;ancestor;ancestor=ancestor.parentElement) ancestors.push(ancestor);
    for(const ancestor of ancestors.reverse()) {
      if(ancestor.tagName==='DETAILS') ancestor.open=true;
      if(ancestor.hidden && ancestor.getAttribute('role')==='tabpanel') {
        document.getElementById(ancestor.getAttribute('aria-labelledby'))?.click();
      }
    }
  }
  async function followView(id, query, requestedHash) {
    const params=new URLSearchParams(query);
    const kind=id==='figure-1'||id.startsWith('roc-title-')?'roc':
      id==='table-1'||id.startsWith('table-title-')?'table':
      id==='figure-2'||id.startsWith('resource-title-')?'resource':null;
    const allowed=kind==='roc'?['dataset','models','prompts','inputs']:['dataset','group'];
    if(!kind || [...params.keys()].some(key=>!allowed.includes(key)||params.getAll(key).length!==1)) {
      viewMessage('This report link contains unsupported or repeated view settings.');return;
    }
    const dataset=params.get('dataset');
    const panels=[...main.querySelectorAll('.dataset-panel')].filter(panel=>panel.dataset.datasetKey.split('-')[0]===dataset);
    if(panels.length!==1) {
      viewMessage('Requested dataset unavailable or ambiguous: '+(dataset||'(not specified)'));return;
    }
    const panel=panels[0];
    const prefix={roc:'roc-title',table:'table-title',resource:'resource-title'}[kind];
    const target=panel.querySelector('[id^="'+prefix+'-"]');
    if(!target) {viewMessage('Requested report section is unavailable.');return;}
    revealTarget(target);
    let error;
    try {
      if(kind==='roc') {
        const options={};
        for(const field of ['models','prompts','inputs']) if(params.has(field)) options[field]=params.get(field);
        error=panel.applyRocView?await panel.applyRocView(options):'Requested ROC controls are unavailable.';
      } else if(kind==='resource') {
        error=panel.applyResourceView
          ?await panel.applyResourceView(params.get('group')||'none')
          :'Requested resource grouping controls are unavailable.';
      } else {
        const figure=target.closest('.results-table');
        error=params.has('group')&&figure?.applyTableView
          ?figure.applyTableView(params.get('group')):'Requested table grouping controls are unavailable.';
      }
    } catch {error='The requested report view could not be displayed.';}
    // Rapid navigation must not scroll back to an earlier, slower plot update.
    if(location.hash!==requestedHash) return;
    if(error) {viewMessage(error);return;}
    viewMessage('');
    requestAnimationFrame(()=>{ revealIndex(target); target.scrollIntoView(); });
  }
  function revealHash() {
    const fragment=location.hash.slice(1), separator=fragment.indexOf('?');
    let id;
    try {id=decodeURIComponent(separator<0?fragment:fragment.slice(0,separator));} catch {return;}
    if(separator>=0) {
      void followView(id,fragment.slice(separator+1),location.hash);return;
    }
    const target=document.getElementById(id);
    revealTarget(target);
    viewMessage('');
    if(target && location.hash) {
      requestAnimationFrame(()=>{ revealIndex(target); target.scrollIntoView(); });
    }
  }
  document.addEventListener('click', event => {
    if (event.defaultPrevented || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    const link = event.target.closest?.('a[href]');
    if (!link || link.hasAttribute('download') || (link.target && link.target !== '_self')) return;
    const destination = new URL(link.href, document.baseURI);
    if (destination.origin === location.origin && destination.pathname === location.pathname &&
        destination.search === location.search && destination.hash && destination.hash === location.hash) {
      event.preventDefault();
      revealHash();
    }
  });
  new MutationObserver(rebuild).observe(main, { subtree: true, attributes: true, attributeFilter: ['hidden'] });
  window.addEventListener('scroll', () => {
    if (!scheduled) { scheduled = true; requestAnimationFrame(selectCurrent); }
  }, { passive: true });
  window.addEventListener('resize', selectCurrent);
  window.addEventListener('hashchange', revealHash);
  rebuild();
  revealHash();
})();
