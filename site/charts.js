const study = JSON.parse(document.getElementById('study-data').textContent);
const figureText = WildlifeReport.theme.text;
const figureTheme = WildlifeReport.figureTheme();
const axisTheme = WildlifeReport.axisTheme();
const palette = ['#70b7ff','#ffa965','#c1a0ff','#5bd4bc','#ff97bb','#d6c578','#7ed3ed','#dba2e9'];
function wrapExportText(text, width, fontSize) {
  // Measure the exported font so long captions fit at the chosen image width.
  const context=document.createElement('canvas').getContext('2d');
  context.font=fontSize+'px Arial';
  const lines=[''];
  for (const word of text.split(/\s+/)) {
    const last=lines.length-1, candidate=lines[last] ? lines[last]+' '+word : word;
    if (context.measureText(candidate).width>width && lines[last]) lines.push(word);
    else lines[last]=candidate;
  }
  return lines;
}
function exportCaptionLines(caption, width, fontSize) {
  // Read the rendered caption, retaining emphasis and paragraph boundaries.
  const context=document.createElement('canvas').getContext('2d');
  const escape=text=>text.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
  const lines=[];
  let line='', length=0, space=false;
  function flush(paragraph=false) {
    if(line) {lines.push(line);line='';length=0;}
    if(paragraph && lines.length && lines.at(-1)!=='') lines.push('');
    space=false;
  }
  function word(text, bold, italic) {
    context.font=(italic?'italic ':'')+(bold?'bold ':'')+fontSize+'px Arial';
    const gap=space&&line?' ':'';
    if(line && length+context.measureText(gap+text).width>width) flush();
    function append(fragment) {
      if(space && line) {line+=' ';length+=context.measureText(' ').width;}
      let markup=escape(fragment);
      if(italic) markup='<i>'+markup+'</i>';
      if(bold) markup='<b>'+markup+'</b>';
      line+=markup;length+=context.measureText(fragment).width;space=false;
    }
    // Split unusually long unbroken text rather than clipping it in an image.
    let fragment='';
    for(const character of text) {
      if(fragment && context.measureText(fragment+character).width>width) {
        append(fragment);flush();fragment='';
      }
      fragment+=character;
    }
    append(fragment);
  }

  function visit(node, bold=false, italic=false) {
    if(node.nodeType===3) {
      for(const token of node.textContent.split(/(\s+)/u)) {
        if(!token) continue;
        if(/^\s+$/u.test(token)) space=true;
        else word(token,bold,italic);
      }
      return;
    }
    if(node.hidden || node.getAttribute?.('aria-hidden')==='true') return;
    const tag=node.tagName?.toLowerCase();
    const block=['p','div','li','dt','dd'].includes(tag);
    if(block) flush(true);
    if(tag==='br') flush();
    for(const child of node.childNodes) visit(child,bold||tag==='strong'||tag==='b',
      italic||tag==='em'||tag==='i');
    if(block) flush(true);
  }
  visit(caption);
  flush();
  while(lines.at(-1)==='') lines.pop();
  return lines;
}
function addExportGuide(layout, heading, caption, dataset, width) {
  const escape=text=>text.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
  const titles=wrapExportText(heading,width-48,18);
  const identity=dataset?wrapExportText(dataset,width-48,12):[];
  const notes=exportCaptionLines(caption,width-48,13);
  const top=titles.length*24+identity.length*18+24;
  const bottom=notes.length*18+40;
  const baseHeight=layout.height;
  const plotHeight=baseHeight-layout.margin.t-layout.margin.b;
  const plotWidth=width-layout.margin.l-layout.margin.r;
  // Preserve the plot and legend geometry: add title/caption space outside it.
  const annotation={text:notes.join('<br>'),showarrow:false,align:'left',
    xref:'paper',yref:'paper',x:(24-layout.margin.l)/plotWidth,
    y:-(layout.margin.b+16)/plotHeight,xanchor:'left',yanchor:'top',
    font:{family:'Arial',size:13,color:figureText}};
  layout.annotations=[...(layout.annotations||[]),annotation];
  layout.margin={...layout.margin,t:layout.margin.t+top,b:layout.margin.b+bottom,autoexpand:false};
  layout.title={text:titles.map(escape).join('<br>')+
    (identity.length?'<br><span style="font-size:12px">'+identity.map(escape).join('<br>')+'</span>':''),
    font:{family:'Arial',size:18,color:figureText},x:0,xref:'container',
    y:1,yref:'container',yanchor:'top',pad:{l:24,t:16}};
  layout.width=width;layout.height=baseHeight+top+bottom;
  return top+bottom;
}
function draw(panel, key) {
  const rows = study[key].rows;
  const chart = panel.querySelector('.chart');
  const legendBox = panel.querySelector('.plot-legend');
  const references = [
    {id:'random',name:'Random review (expected)',color:'#94a3b8'},
    {id:'recall',name:'95% TPR reference',color:'#f08b9b'}
  ];
  const visibleReferences = new Set(references.map(r=>r.id));
  let dragMode = 'zoom';
  let plotReady = Promise.resolve();
  const plottable = rows.filter(r => r.point || (r.metrics &&
    r.metrics.positive_recall !== null &&
    r.metrics.false_positive_rate !== null &&
    ['complete','finished with failures'].includes(r.status)));
  const colorFor = row => {
    if (row.kind === 'published') {
      if (row.model.startsWith('ACTL ')) return '#70b7ff';
      if (row.model.startsWith('LTS')) return '#ffa965';
    }
    return palette[plottable.findIndex(r => r.model === row.model) % palette.length];
  };
  const siteControls = WildlifeReport.createSelection(panel,key,rows,plottable,()=>render());
  const selectedRows = () => siteControls.rows();
  const rowLabel = row => WildlifeReport.seriesLabel(row);
  function traces() {
    const output = [];
    for (const row of selectedRows()) {
      const color = colorFor(row);
      const coverageNote = row.kind === 'measured'
        ? `<br>Coverage ${row.valid_predictions}/${row.metrics.n} ` +
          `(${(row.coverage*100).toFixed(1)}%)` +
          ` · ${row.failures} failures (always reviewed)` : '';
      if (row.point) {
        const symbol = row.kind === 'published'
          ? (row.input_modality === 'text-image' ? 'square' : 'x') : 'diamond';
        output.push({x:[row.point[0]*100],y:[row.point[1]*100],mode:'markers',
          name:rowLabel(row),marker:{size:13,symbol,color},cliponaxis:row.valid_predictions!==0,
          hovertemplate:'%{x:.1f}% FPR · %{y:.1f}% TPR' +
            coverageNote + '<extra>%{fullData.name}</extra>'});
      } else if (row.points && row.points.length) {
        output.push({x:row.points.map(p=>p[0]*100),y:row.points.map(p=>p[1]*100),
          customdata:row.thresholds,legendgroup:row.series,
          mode:'lines',line:{shape:'hv',width:2.4,color,dash:row.prompt==='few-shot'?'dash':'solid'},
          name:rowLabel(row),hovertemplate:'%{x:.1f}% FPR · %{y:.1f}% TPR' +
            '<br>Cutoff %{customdata}' +
            coverageNote + '<extra>%{fullData.name}</extra>'});
        output.push({x:[row.metrics.false_positive_rate*100],
          y:[row.metrics.positive_recall*100],
          mode:'markers',name:rowLabel(row),showlegend:false,legendgroup:row.series,
          marker:{size:9,color,line:{color:'white',width:1}},
          hovertemplate:'Fixed 0.5 cutoff<br>%{x:.1f}% FPR · %{y:.1f}% TPR' +
            coverageNote + '<extra>%{fullData.name}</extra>'});
      } else if (row.metrics) {
        output.push({x:[row.metrics.false_positive_rate*100],
          y:[row.metrics.positive_recall*100],
          mode:'markers',name:rowLabel(row),marker:{size:12,color},
          hovertemplate:'%{x:.1f}% FPR · %{y:.1f}% TPR' +
            coverageNote + '<extra>%{fullData.name}</extra>'});
      }
    }
    if (visibleReferences.has('random')) output.push({
      x:[0,100],y:[0,100],mode:'lines',name:references[0].name,
      line:{color:references[0].color,dash:'dot',width:1.5}});
    if (visibleReferences.has('recall')) output.push({
      x:[0,100],y:[95,95],mode:'lines',name:references[1].name,
      line:{color:references[1].color,dash:'dot',width:1.5}});
    return output;
  }
  function renderLegend() {
    siteControls.legend(legendBox,selectedRows(),colorFor,references);
  }
  function plotLayout() {
    return {...figureTheme,margin:{l:95,r:20,t:25,b:90},
      xaxis:{...axisTheme,title:{
        text:'Irrelevant adverts sent for review<br>(false-positive rate, FPR, %)',
        standoff:12,font:{size:14,color:figureText}},automargin:true,range:[0,100]},
      yaxis:{...axisTheme,title:{text:'Relevant adverts found (TPR, %)',
        standoff:12,font:{size:14,color:figureText}},automargin:true,range:[0,100]},
      showlegend:false,dragmode:dragMode,uirevision:key,hovermode:'closest'};
  }
  function renderPlot() {
    plotReady=Plotly.react(chart,traces(),plotLayout(),
      {responsive:true,displayModeBar:false,scrollZoom:false,displaylogo:false});
  }
  function render() { renderLegend(); renderPlot(); }
  function setDragMode(mode) {
    dragMode=mode;
    panel.querySelector('.zoom').setAttribute('aria-pressed',String(mode==='zoom'));
    panel.querySelector('.pan').setAttribute('aria-pressed',String(mode==='pan'));
    plotReady=plotReady.then(()=>Plotly.relayout(chart,{dragmode:mode}));
  }
  panel.querySelector('.zoom').onclick=()=>setDragMode('zoom');
  panel.querySelector('.pan').onclick=()=>setDragMode('pan');
  panel.querySelector('.reset').onclick=()=>{
    plotReady=plotReady.then(()=>Plotly.relayout(chart,{
      'xaxis.range':[0,100],'yaxis.range':[0,100],
      'xaxis.autorange':false,'yaxis.autorange':false}));
  };
  async function download(format) {
    await plotReady;
    const figure=document.createElement('div');
    figure.style.cssText='position:fixed;left:-10000px;top:0';
    document.body.append(figure);
    const exportTraces=traces();
    const legendCount=exportTraces.filter(trace=>trace.showlegend!==false).length;
    const extraHeight=Math.max(80,legendCount*27+40);
    const width=Math.max(1000,chart.clientWidth), height=chart.clientHeight;
    const layout=plotLayout();
    layout.margin={...layout.margin,b:layout.margin.b+extraHeight,autoexpand:false};
    layout.height=height+extraHeight;
    addExportGuide(layout,
      panel.querySelector('[id^="roc-title-"]').textContent,
      panel.querySelector('.roc-figure figcaption'),
      document.getElementById(panel.getAttribute('aria-labelledby')).textContent,width);

    layout.showlegend=true;
    layout.legend={orientation:'v',x:0,y:-0.15,yanchor:'top',
      traceorder:'normal',font:{size:11}};
    layout.xaxis.range=[...chart._fullLayout.xaxis.range];
    layout.yaxis.range=[...chart._fullLayout.yaxis.range];
    layout.xaxis.autorange=false; layout.yaxis.autorange=false;
    const exportHeight=layout.height;
    try {
      await Plotly.newPlot(figure,exportTraces,layout,
        {staticPlot:true,displayModeBar:false,responsive:false});
      await Plotly.downloadImage(figure,{format,filename:key+'-roc',
        width,height:exportHeight});
    } finally {
      Plotly.purge(figure); figure.remove();
    }
  }
  panel.applyRocView=async options=>{
    const error=siteControls.applyView(options);
    if(error) return error;
    await plotReady;
    await Plotly.relayout(chart,{'xaxis.range':[0,100],'yaxis.range':[0,100],
      'xaxis.autorange':false,'yaxis.autorange':false});
    return null;
  };
  panel.querySelector('.png').onclick=()=>download('png');
  panel.querySelector('.svg').onclick=()=>download('svg');
  WildlifeReport.initializeResources(panel,study[key].resource_views,Plotly,addExportGuide,key);
  render();
}
document.querySelectorAll('.dataset-panel').forEach(panel=>{
  if(panel.querySelector('.chart')) draw(panel,panel.dataset.datasetKey);
});
WildlifeReport.initializeTabs();
WildlifeReport.initializePromptViews();
WildlifeReport.initializeTableGrouping();
