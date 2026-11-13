/* Website-only ROC selection and independent accessible tab groups. */
((root) => {
  // Shared with ROC, summary and image exports; return fresh layouts for Plotly.
  const theme = Object.freeze({background:'#202222',text:'#ededed',muted:'#abadaf',
    grid:'#414444',zero:'#777979',hover:'#303232',highlight:'#303232'});
  const figureTheme = () => ({paper_bgcolor:theme.background,plot_bgcolor:theme.background,
    font:{color:theme.text},hoverlabel:{bgcolor:theme.hover,font:{color:theme.text}}});
  const axisTheme = () => ({color:theme.text,gridcolor:theme.grid,zerolinecolor:theme.zero});
  const modelOrder = ['gpt_6_luna_chat', 'gpt_6_luna_decisions', 'microsoft_decision_1', 'd3_27b_q4_k_m'];
  const modelNames = {
    gpt_6_luna_chat: 'GPT-6 Luna Chat',
    gpt_6_luna_decisions: 'GPT-6 Luna Decisions',
    microsoft_decision_1: 'Microsoft-Decision-1',
    d3_27b_q4_k_m: 'd3'
  };
  const inputNames = {
    title: 'Title', title_and_desc: 'title + desc',
    combined_text: 'title + desc'
  };
  const promptNames = {'zero-shot': 'Zero-shot', 'few-shot': 'Few-shot'};
  const baselineNames = {
    'ACTL text': 'ACTL · Text', 'LTS-text': 'LTS · Text',
    'ACTL text-image': 'ACTL · Text & Image', 'LTS text-image': 'LTS · Text & Image',
    'GPT-4 zero-shot': 'GPT-4'
  };
  const labelFor = (kind, value) => ({model: modelNames, input: inputNames, prompt: promptNames}[kind]?.[value] || value);
  // Input storage keys remain intact; controls select the same displayed evidence type.
  const sameOption = (field,a,b) => field==='input' ? labelFor(field,a)===labelFor(field,b) : a===b;
  const includesOption = (selection,field,value) => selection[field].some(item=>sameOption(field,item,value));
  function seriesLabel(row) {
    if (!row) return 'Unknown configuration';
    if (row.kind === 'published') return baselineNames[row.model] || row.model;
    return [labelFor('model', row.model), labelFor('prompt', row.prompt), labelFor('input', row.input)].join(' · ');
  }
  function plottableRows(rows) {
    return rows.filter(row => row.point || (row.metrics &&
      row.metrics.positive_recall !== null && row.metrics.false_positive_rate !== null &&
      ['complete', 'finished with failures'].includes(row.status)));
  }
  function defaultSelection(key, rows) {
    const task = key.split('-')[0];
    const input = task === 'sharks' ? 'title_and_desc' : task === 'birds' ? 'combined_text' : 'title';
    const available = plottableRows(rows);
    const candidate = available.find(row => row.kind === 'measured' &&
      row.model === 'd3_27b_q4_k_m' && row.prompt === 'zero-shot' && row.input === input);
    return {
      model: candidate ? [candidate.model] : [],
      prompt: candidate ? [candidate.prompt] : [],
      input: candidate ? [candidate.input] : [],
      baselines: available.filter(row => row.kind === 'published').map(row => row.series)
    };
  }
  function selectRows(rows, selection) {
    return rows.filter(row => row.kind === 'published'
      ? selection.baselines.includes(row.series)
      : ['model', 'prompt', 'input'].every(field => includesOption(selection,field,row[field])));
  }
  function availableValues(rows, field) {
    const values = [...new Set(rows.filter(row => row.kind === 'measured').map(row => row[field]))];
    const preferred = field === 'model' ? modelOrder : field === 'prompt'
      ? ['zero-shot', 'few-shot'] : ['title', 'title_and_desc', 'combined_text'];
    const ordered=[...preferred.filter(value => values.includes(value)), ...values.filter(value => !preferred.includes(value))];
    return ordered.filter((value,index)=>ordered.findIndex(other=>sameOption(field,other,value))===index);
  }
  function linkedRocSelection(available, selection, options) {
    // Validate the complete requested view before changing any existing checkbox.
    const next={...selection};
    const fields={models:'model',prompts:'prompt',inputs:'input'};
    for(const [parameter,field] of Object.entries(fields)) {
      if(options[parameter]===undefined) continue;
      let wanted=options[parameter].split(',');
      if(parameter==='models' && options.models==='all') wanted=[...modelOrder];
      if(parameter==='models' && options.models==='decision') wanted=modelOrder.filter(m=>m!=='gpt_6_luna_chat');
      const choices=availableValues(available,field);
      const matched=wanted.map(value=>choices.find(choice=>sameOption(field,choice,value)));
      if(matched.some(value=>value===undefined)) return {error:'Requested '+parameter+' unavailable: '+wanted.filter((value,i)=>matched[i]===undefined).join(', ')};
      next[field]=[...new Set(matched)];
    }
    for(const model of next.model) for(const prompt of next.prompt) for(const input of next.input) {
      if(!available.some(row=>row.kind==='measured' && row.model===model && row.prompt===prompt && sameOption('input',row.input,input))) {
        return {error:'Requested ROC configuration unavailable: '+labelFor('model',model)+' / '+labelFor('prompt',prompt)+' / '+labelFor('input',input)};
      }
    }
    if(!next.model.length || !next.prompt.length || !next.input.length) return {error:'The requested ROC view has no measured configurations.'};
    return {selection:next};
  }
  function createSelection(panel, key, rows, available, onchange) {
    const selection = defaultSelection(key, rows);
    const defaultMissing = !selection.model.length;
    const host = document.createElement('div');
    host.className = 'roc-selection';
    const columns = document.createElement('div');
    columns.className = 'selection-columns';
    function box(field, value, text, disabled = false) {
      const label = document.createElement('label'); label.className = 'selection-box';
      const checkbox = document.createElement('input'); checkbox.type = 'checkbox';
      checkbox.dataset.selectionField = field;
      checkbox.value = value; checkbox.checked = includesOption(selection,field,value); checkbox.disabled = disabled;
      checkbox.addEventListener('change', () => {
        selection[field] = checkbox.checked ? [...selection[field], value] : selection[field].filter(item => !sameOption(field,item,value));
        onchange();
      });
      const title = document.createElement('span'); title.textContent = text;
      label.append(checkbox, title); return label;
    }
    for (const [field, title] of [['model', 'Model'], ['prompt', 'Prompt'], ['input', 'Input']]) {
      const group = document.createElement('fieldset');
      const legend = document.createElement('legend'); legend.textContent = title;
      group.append(legend, ...availableValues(available, field).map(value => box(field, value, labelFor(field, value))));
      columns.append(group);
    }
    host.append(columns);
    const baselines = document.createElement('fieldset'); baselines.className = 'baseline-selection';
    const legend = document.createElement('legend'); legend.textContent = 'Baselines'; baselines.append(legend);
    const baselineRows = rows.filter(row => row.kind === 'published');
    if (!baselineRows.length) {
      const note = document.createElement('p'); note.textContent = 'No published ROC baselines are available for this dataset.';
      baselines.append(note);
    }
    for (const row of baselineRows) {
      const disabled = !available.some(candidate => candidate.series === row.series);
      const reason = disabled ? (row.metrics?.positive_f1 != null || row.model.startsWith('GPT-4')
        ? ' · F1 only; shown in table' : ' · ROC unavailable; shown in table') : '';
      baselines.append(box('baselines', row.series, seriesLabel(row) + reason, disabled));
    }
    const message = document.createElement('p'); message.className = 'selection-message'; message.setAttribute('role', 'status');
    host.append(baselines, message);
    panel.querySelector('.plot-tools').insertAdjacentElement('beforebegin', host);
    return {
      applyView(options) {
        const proposed=linkedRocSelection(available,selection,options);
        if(proposed.error) return proposed.error;
        Object.assign(selection,proposed.selection);
        for(const checkbox of host.querySelectorAll('input')) {
          checkbox.checked=includesOption(selection,checkbox.dataset.selectionField,checkbox.value);
        }
        onchange();
        return null;
      },
      rows() {
        const selected = selectRows(available, selection);
        message.textContent = selected.some(row => row.kind === 'measured') ? '' : defaultMissing && !selection.model.length
          ? 'The requested d3 zero-shot default is unavailable. Select a model, prompt and input to display measured results.'
          : 'No measured configurations match these selections.';
        return selected;
      },
      legend(container, selected, colorFor, references) {
        container.replaceChildren();
        function entry(text, color, dash, marker) {
          const item = document.createElement('div'); item.className = 'readonly-legend-item';
          const swatch = document.createElement('span');
          swatch.className = marker ? 'legend-marker' : 'legend-swatch';
          if (marker) { swatch.textContent = marker; swatch.style.color = color; }
          else { swatch.style.borderColor = color; if (dash) swatch.style.borderStyle = 'dashed'; }
          const label = document.createElement('span'); label.textContent = text;
          item.append(swatch, label); container.append(item);
        }
        for (const row of selected) entry(seriesLabel(row), colorFor(row), row.prompt === 'few-shot',
          row.kind === 'published' ? (row.input_modality === 'text-image' ? '■' : '×') : row.point ? '◆' : null);
        for (const reference of references) entry(reference.name, reference.color, true, null);
      }
    };
  }
  function initializeTabs() {
    const lists = [...document.querySelectorAll('.report-page [role="tablist"]')];
    for (const list of lists) {
      const tabs = [...list.querySelectorAll('[role="tab"]')];
      function activate(tab, focus = false) {
        for (const button of tabs) {
          const selected = button === tab;
          button.setAttribute('aria-selected', String(selected)); button.tabIndex = selected ? 0 : -1;
          document.getElementById(button.getAttribute('aria-controls')).hidden = !selected;
        }
        if (focus) tab.focus();
        const panel = document.getElementById(tab.getAttribute('aria-controls'));
        requestAnimationFrame(() => panel.querySelectorAll('.chart, .resource-chart').forEach(chart => {
          if (chart.data) root.Plotly.Plots.resize(chart);
        }));
      }
      tabs.forEach((tab, index) => {
        tab.addEventListener('click', () => activate(tab));
        tab.addEventListener('keydown', event => {
          let next;
          if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
          if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
          if (event.key === 'Home') next = 0;
          if (event.key === 'End') next = tabs.length - 1;
          if (next !== undefined) { event.preventDefault(); activate(tabs[next], true); tabs[next].scrollIntoView({block:'nearest', inline:'nearest'}); }
        });
      });
      if (tabs.length) activate(tabs.find(tab => tab.getAttribute('aria-selected') === 'true') || tabs[0]);
    }
  }
  function promptSelection(records, family = 'decision', condition = 'zero-shot', hash = null) {
    const families = ['decision','llm'].filter(value => records.some(row => row.family === value));
    const selectedFamily = families.includes(family) ? family : families[0] || family;
    const conditions = ['zero-shot','few-shot'].filter(value => records.some(row =>
      row.family === selectedFamily && row.condition === value));
    const selectedCondition = conditions.includes(condition) ? condition : conditions[0] || condition;
    const candidates = records.filter(row => row.family === selectedFamily && row.condition === selectedCondition);
    const selected = candidates.find(row => row.sha256 === hash) || candidates[0] || null;
    return {family:selectedFamily,condition:selectedCondition,families,conditions,candidates,selected};
  }
  function initializePromptViews() {
    for (const viewer of document.querySelectorAll('.prompt-viewer')) {
      const cards = [...viewer.querySelectorAll('.prompt-source')];
      const records = cards.map(card => ({...card.dataset, card}));
      const versionHost = viewer.querySelector('.prompt-version');
      const version = versionHost.querySelector('select');
      const message = viewer.querySelector('.prompt-message');
      const remembered = new Map();
      let family='decision', condition='zero-shot';
      function update() {
        const state = promptSelection(records,family,condition,remembered.get(family+'/'+condition));
        family=state.family; condition=state.condition;
        for (const input of viewer.querySelectorAll('.prompt-family input')) {
          input.checked=input.value===family;
          input.disabled=!state.families.includes(input.value);
          input.closest('label').title=input.disabled ? 'Not recorded for this dataset' : '';
        }
        for (const input of viewer.querySelectorAll('.prompt-condition input')) {
          input.checked=input.value===condition;
          input.disabled=!state.conditions.includes(input.value);
          input.closest('label').title=input.disabled ? 'Not recorded for this dataset and model type' : '';
        }
        versionHost.hidden=state.candidates.length<2;
        version.replaceChildren(...state.candidates.map(row=>{
          const option=document.createElement('option');option.value=row.sha256;
          option.textContent=row.version+' · '+row.sha256.slice(0,8);return option;
        }));
        for (const row of records) row.card.hidden=row!==state.selected;
        if (state.selected) {
          version.value=state.selected.sha256;
          remembered.set(family+'/'+condition,state.selected.sha256);
        }
        message.textContent=state.selected ? '' : 'No recorded prompt definitions are available for this dataset.';
        message.hidden=Boolean(state.selected);
      }
      viewer.querySelectorAll('input[type="radio"]').forEach(input=>input.addEventListener('change',()=>{
        if (input.closest('.prompt-family')) family=input.value; else condition=input.value;
        update();
      }));
      version.addEventListener('change',()=>{remembered.set(family+'/'+condition,version.value);update();});
      update();
    }
  }
  function initializeTableGrouping(document = root.document) {
    for (const figure of document.querySelectorAll('.results-table')) {
      if (figure.dataset.groupingReady) continue;
      const data = figure.querySelector('.table-grouping-data');
      if (!data) continue;
      const views = JSON.parse(data.textContent);
      const table = figure.querySelector('table');
      const caption = table.querySelector('caption').outerHTML;
      const controls = [...figure.querySelectorAll('.table-grouping input')];
      if (!controls.length) continue;
      const status = figure.querySelector('.grouping-status');
      function update() {
        const selected = controls.filter(input => input.checked).map(input => input.value);
        table.innerHTML = caption + views[selected.join(',') || 'none'];
        status.textContent = selected.length
          ? 'Grouped by '+selected.map(value => value==='input'?'Input':'Prompt').join(' and ')+'.'
          : '';
      }
      figure.applyTableView = group => {
        const fields=group==='none'?[]:group.split(',');
        if(fields.some(field=>!controls.some(control=>control.value===field))) return 'Requested table grouping unavailable: '+group;
        controls.forEach(control=>control.checked=fields.includes(control.value));
        update();
        return null;
      };
      controls.forEach(input => input.addEventListener('change', update));
      figure.dataset.groupingReady = 'true';
    }
  }
  function resourcePlotSpec(view) {
    const grouped=view.fields.length>0, rows=view.rows;
    const names=rows.map(row=>[labelFor('model',row.model),
      ...['prompt','input'].filter(field=>!view.fields.includes(field))
        .map(field=>labelFor(field,row[field]))].join(' · '));
    const ids=rows.map(row=>row.series);
    const axes=axisTheme();
    const annotations=[
      {xref:'paper',yref:'paper',x:0.19,y:1,yshift:16,yanchor:'bottom',
        text:'API charge<br>/ 1,000 adverts (USD)',xanchor:'center',showarrow:false,font:{size:13,color:theme.text}},
      {xref:'paper',yref:'paper',x:0.75,y:1,yshift:16,yanchor:'bottom',
        text:'Summed request time<br>/ 1,000 adverts (s)',xanchor:'center',showarrow:false,font:{size:13,color:theme.text}}
    ];
    const traces=[],ranges=[];
    for (const [index,prefix,metric,precision,color] of [
      [0,'cost','cost_per_advert',5,'#70b7ff'],
      [1,'latency','latency_per_advert',0,'#ffa965']]) {
      const values=rows.map(row=>row[metric]===null?null:row[metric]*1000);
      const deviations=rows.map(row=>row[prefix+'_sd']===null?null:row[prefix+'_sd']*1000);
      const details=rows.map((row,i)=>[names[i],deviations[i],row.count,
        'Input: '+[...new Set(row.inputs.map(value=>labelFor('input',value)))].join('; ')
        +'<br>Prompt: '+row.prompts.map(value=>labelFor('prompt',value)).join('; '),
        row[prefix+'_availability']]);
      rows.forEach((row,i)=>{
        let text=values[i]===null?'N/A':index===0&&row.local_api_free?'0 external API':values[i].toFixed(precision);
        if(grouped&&values[i]!==null) text+=' (±'+(deviations[i]===0?'0':deviations[i].toFixed(precision))+')';
        annotations.push({xref:'paper',yref:'y',x:index===0?0.43:0.99,y:ids[i],xanchor:'right',
          text,showarrow:false,font:{size:10,color:theme.text},
          hovertext:row[prefix+'_availability']});
      });
      const trace={type:'bar',orientation:'h',x:values,y:ids,xaxis:index===0?'x':'x2',
        marker:{color},name:index===0?'API charge':'Summed request time',customdata:details,
        hovertemplate:(index===0?'USD ':'')+'%{x:.'+precision+'f} '+
          (index===0?'':'request-seconds ')+'/ 1,000 adverts'+
          (grouped?'<br>Population SD: %{customdata[1]:.'+precision+'f}'+
            '<br>%{customdata[2]} configs<br>%{customdata[3]}':'')+
          '<br>%{customdata[4]}<extra>%{customdata[0]}</extra>'};
      if(grouped) trace.error_x={type:'data',array:deviations,symmetric:true,visible:true,
        color:theme.text,thickness:1.3,width:4};
      traces.push(trace);
      const ends=values.flatMap((value,i)=>value===null?[]:
        [value-(grouped?deviations[i]:0),value+(grouped?deviations[i]:0)]);
      const low=Math.min(0,...ends),high=Math.max(0,...ends);
      // Leave room for mean/SD annotations beyond the complete whiskers.
      ranges.push([low<0?low-(high-low)*0.05:0,high>0?high*(grouped?2:1.35):1]);
    }
    // Row-major annotation order preserves the ungrouped reading order.
    const cells=annotations.splice(2);
    rows.forEach((row,i)=>annotations.push(cells[i],cells[rows.length+i]));
    const layout={height:Math.max(270,rows.length*38+130),margin:{l:230,r:20,t:55,b:45},
      showlegend:false,barmode:'overlay',bargap:0.35,annotations,
      xaxis:{...axes,domain:[0,0.38],range:ranges[0],rangemode:'tozero',tickformat:'.3~f',fixedrange:true,zeroline:true},
      xaxis2:{...axes,domain:[0.56,0.94],range:ranges[1],rangemode:'tozero',fixedrange:true,zeroline:true},
      yaxis:{...axes,categoryorder:'array',categoryarray:ids,tickvals:ids,ticktext:names,
        autorange:'reversed',automargin:true,fixedrange:true},hovermode:'closest',
      ...figureTheme()};
    return {traces,layout};
  }
  function initializeResources(panel,views,Plotly,addExportGuide,key,document=root.document) {
    const chart=panel.querySelector('.resource-chart');
    if(!views.none.rows.length) {
      chart.textContent='No completed pinned arms with measured request resources.';
      panel.querySelector('.resource-exports').hidden=true;
      return;
    }
    const controls=[...panel.querySelectorAll('.resource-grouping input')];
    const status=panel.querySelector('.resource-grouping-status');
    let selected='none',ready=Promise.resolve();
    function render() {
      const fields=controls.filter(input=>input.checked).map(input=>input.value);
      selected=fields.join(',')||'none';
      const {traces,layout}=resourcePlotSpec(views[selected]);
      chart.style.height=layout.height+'px';
      ready=Plotly.react(chart,traces,layout,{responsive:true,displayModeBar:false,scrollZoom:false,displaylogo:false});
      if(status) status.textContent=fields.length?'Grouped by '+fields.map(value=>value==='input'?'Input':'Prompt').join(' and ')+'.':'';
    }
    controls.forEach(input=>input.addEventListener('change',render));
    panel.applyResourceView=async group=>{
      const fields=group==='none'?[]:group.split(',');
      if(fields.some(field=>!controls.some(control=>control.value===field))) {
        return 'Requested resource grouping unavailable: '+group;
      }
      controls.forEach(control=>{control.checked=fields.includes(control.value)});
      render();
      await ready;
      return null;
    };
    async function download(format) {
      const view=views[selected];
      await ready;
      const figure=document.createElement('div');
      figure.style.cssText='position:fixed;left:-10000px;top:0';document.body.append(figure);
      const {traces,layout}=resourcePlotSpec(view);
      const width=Math.max(1200,chart.clientWidth);
      const grouping=view.fields.length?' Grouped across '+view.fields.map(value=>value==='input'?'Input':'Prompt').join(' and ')+'.':'';
      addExportGuide(layout,panel.querySelector('[id^="resource-title-"]').textContent,
        panel.querySelector('.resource-panel figcaption'),
        document.getElementById(panel.getAttribute('aria-labelledby')).textContent+grouping,width);

      const exportHeight=layout.height;
      try {
        await Plotly.newPlot(figure,traces,layout,{staticPlot:true,displayModeBar:false,responsive:false});
        await Plotly.downloadImage(figure,{format,filename:key+'-resources',width,height:exportHeight});
      } finally {Plotly.purge(figure);figure.remove();}
    }
    panel.querySelector('.resource-png').onclick=()=>download('png');
    panel.querySelector('.resource-svg').onclick=()=>download('svg');
    render();
  }
  const api = {theme, figureTheme, axisTheme, labelFor, seriesLabel, plottableRows, defaultSelection, selectRows, availableValues, linkedRocSelection, createSelection, initializeTabs, promptSelection, initializePromptViews, initializeTableGrouping, resourcePlotSpec, initializeResources};
  root.WildlifeReport = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(globalThis);
