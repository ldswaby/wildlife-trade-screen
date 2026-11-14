/* Cross-dataset presentation, independent of dataset and ROC filter state. */
((root) => {
  const report = root.WildlifeReport || require('./report.js');
  const theme = report.theme;
  const colors = ['#70b7ff', '#ffa965', '#5bd4bc'];
  const titles = ['Balanced accuracy (%) ↑', 'FPR @ 95% TPR (%) ↓',
    'API charge / 1,000<br>adverts (USD) ↓',
    'Summed request time<br>/ 1,000 adverts (s) ↓'];
  const inputNames={title:'title',title_and_desc:'title + desc',combined_text:'title + desc'};
  // Keep one ordered field contract across measured and published point types.
  function hover(metric, models, datasets, value, {mean=false,sd=null,basis='',notes='',points=[]}={}) {
    const escape=text=>String(text).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    const wrap=text=>escape(text).replace(/(.{1,76})(?:\s+|$)/g,'$1<br>').replace(/<br>$/,'');
    const field=(name,text)=>'<b>'+name+':</b> '+wrap(text);
    const unit=metric.key==='balanced_accuracy'||metric.key==='fpr_at_95_tpr'?'%':
      metric.key==='cost_per_advert'?' USD':' s';
    const number=v=>Number(v.toPrecision(6)).toString();
    const fields=[field('Metric',metric.label.replace(/ [↑↓]$/,'')),
      field('Model(s)',models.join(', ')),field('Dataset(s)',datasets.join(', '))];
    if(points.length) {
      fields.push(field('Input(s)',[...new Set(points.map(p=>inputNames[p.input]||p.input))].join(', ')),
        field('Prompt(s)','zero-shot'));
    }
    fields.push(field(mean?'Mean':'Value',number(value)+unit));
    if(sd!==null) {
      fields.push(field('Standard deviation',number(sd)+(unit==='%'?' pp':unit)));
      notes='Population SD '+basis+'. '+notes;
    }
    if(notes) fields.push(field('Notes',notes));
    return fields.join('<br>');
  }
  function inputNotes(points) {
    return points.map(p=>p.dataset+': '+(inputNames[p.input]||p.input)).join('; ');
  }
  function plotSpec(summary) {
    const traces = [], annotations = [];
    const split=summary.baselines.split_policy_points;
    const splitSummary=summary.baselines.split_policy_summary;
    const baselineRow=summary.metrics[0].cells.length;
    const coverage=cell=>cell.component_total
      ?cell.component_n+'/'+cell.component_total+' model–dataset values; '+cell.n+'/3 complete dataset averages'
      :cell.n+'/3 datasets';
    const layout = {height:490+(baselineRow-4)*60, dragmode:false, margin:{l:190,r:22,t:90,b:95},
      ...report.figureTheme(),font:{color:theme.text,family:'Arial'},
      legend:{orientation:'h',x:0.5,xanchor:'center',y:-0.2,itemclick:false,itemdoubleclick:false},
      yaxis:{range:[baselineRow+0.5,-0.5],tickvals:Array.from({length:baselineRow+1},(_,i)=>i),
        ticktext:[...summary.metrics[0].cells.map(c=>c.component_total
          ?'<b>Decision models</b> (<i>μ</i>)':c.name),'Supervised baselines'],
        ticklabelstandoff:34,fixedrange:true,
        showgrid:false,zeroline:false},annotations,
      shapes:[{type:'rect',xref:'paper',yref:'y',x0:0,x1:1,y0:baselineRow-0.5,y1:baselineRow+0.5,
        fillcolor:theme.highlight,opacity:1,line:{width:0},layer:'below'}]};
    const aggregateRow=summary.metrics[0].cells.findIndex(cell=>cell.component_total);
    if(aggregateRow>=0) {
      const line={color:'#707575',width:1.1};
      const bracket=(x0,y0,x1,y1)=>({type:'line',xref:'paper',yref:'y',x0,y0,x1,y1,line});
      layout.shapes.push(bracket(-.014,-.5,-.014,aggregateRow+.5),
        ...[-.5,aggregateRow-.5,aggregateRow+.5].map(y=>bracket(-.024,y,-.014,y)));
    }
    summary.metrics.forEach((metric,index)=>{
      const suffix=index===0?'':String(index+1), axis='x'+suffix;
      const start=index*0.255, end=start+0.22;
      const values=[];
      for(let dataset=0;dataset<3;dataset++) {
        const x=[],y=[],customdata=[];
        metric.cells.forEach((cell,row)=>{
          const point=cell.points.find(p=>p.dataset===summary.datasets[dataset]);
          if(point) {
            x.push(point.value);y.push(row+point.offset);
            customdata.push(hover(metric,cell.models||[cell.name],[point.dataset],point.value,{
              points:[point],mean:!!cell.component_total,sd:point.sd??null,basis:'across 3 models',
              notes:cell.component_total
                ?'Equal model weights; average individual model, not an ensemble. SD describes variation, not uncertainty.':''
            }));
            values.push(point.value);
          }
        });
        if(index===0) {
          const point=summary.baselines.points.find(p=>p.dataset===summary.datasets[dataset]);
          if(point) {
            x.push(point.value);y.push(baselineRow+point.offset);values.push(point.value);
            customdata.push(hover(metric,[point.model],[point.dataset],point.value,{
              notes:point.caveat
            }));
          }
        }
        traces.push({type:'scatter',mode:'markers',x,y,xaxis:axis,yaxis:'y',
          name:summary.datasets[dataset],legendgroup:String(dataset),showlegend:index===0,cliponaxis:false,
          marker:{size:7,color:colors[dataset],opacity:0.85},customdata,
          hovertemplate:'%{customdata}<extra></extra>'});
      }
      metric.cells.forEach((cell,row)=>{
        if(cell.mean!==null) values.push(cell.mean-cell.sd,cell.mean+cell.sd);
        else annotations.push({xref:axis+' domain',yref:'y',x:0.55,y:cell.n===0?row:row+0.3,
          text:'N/A',showarrow:false,font:{size:11,color:theme.muted},
          hovertext:coverage(cell)+'; '+cell.unavailable});
      });
      traces.push({type:'scatter',mode:'markers',x:metric.cells.map(c=>c.mean),
        y:metric.cells.map((_,row)=>row),xaxis:axis,yaxis:'y',name:'Mean ±1 SD',showlegend:false,cliponaxis:false,
        marker:{size:12,symbol:metric.cells.map(c=>c.component_total?'diamond':'circle'),color:theme.text,opacity:1,line:{width:1,color:theme.background}},
        error_x:{type:'data',array:metric.cells.map(c=>c.sd??0),visible:true,
          color:theme.text,thickness:1.8,width:5},
        customdata:metric.cells.map(c=>c.mean===null?'':hover(metric,c.models||[c.name],
          c.points.map(p=>p.dataset),c.mean,{points:c.points,mean:true,sd:c.sd,
            basis:c.component_total?'across 9 model–dataset values':'across 3 datasets',
            notes:'Equal weights. '+coverage(c)+'. Input: '+inputNotes(c.points)+
              '. SD describes variation, not uncertainty.'+(c.component_total
                ?' Average individual model; not an ensemble.':'')})),
        hovertemplate:'%{customdata}<extra></extra>'});
      if(index===0&&splitSummary) values.push(splitSummary.mean-splitSummary.sd,splitSummary.mean+splitSummary.sd);
      if(index>0 || !(summary.baselines.points.length||splitSummary)) annotations.push({
        xref:axis+' domain',yref:'y',x:0.55,y:baselineRow,text:'N/A',showarrow:false,
        font:{size:11,color:theme.muted},hovertext:index===0?
          [summary.baselines.unavailable,summary.baselines.split_policy_unavailable].filter(Boolean).join('; '):
          'No comparable published metric or resource evidence'});
      let low=values.length?Math.min(...values):0, high=values.length?Math.max(...values):1;
      if(index>0) low=Math.min(0,low);
      const padding=(high-low||Math.abs(high)||1)*0.1;
      low=index>0&&low===0?0:low-padding;high+=padding;
      layout['xaxis'+suffix]={type:'linear',domain:[start,end],range:[low,high],
        fixedrange:true,...report.axisTheme(),
        tickfont:{size:11},ticksuffix:index<2?'%':'',nticks:4,
        tickformat:index===2?'.3~f':index===3?'.3~s':'.1~f'};
      annotations.push({xref:'paper',yref:'paper',x:(start+end)/2,y:1.15,
        text:titles[index],xanchor:'center',yanchor:'bottom',showarrow:false,font:{size:12,color:theme.text}});
    });
    if(splitSummary) traces.push({type:'scatter',mode:'markers',
      x:[splitSummary.mean],y:[baselineRow+splitSummary.offset],xaxis:'x',yaxis:'y',
      name:'Live Birds (split policy)',legendgroup:'birds-split-policy',showlegend:true,
      cliponaxis:false,marker:{symbol:'x-thin',size:9,color:colors[2],opacity:0.85,
        line:{color:colors[2],width:1}},
      error_x:{type:'data',array:[splitSummary.sd],symmetric:true,visible:true,
        color:'rgba(91,212,188,0.85)',thickness:1.8,width:5},
      customdata:[hover(summary.metrics[0],[...new Set(split.map(p=>p.model))],
        split.map(p=>'Live Birds — '+p.policy),splitSummary.mean,{mean:true,sd:splitSummary.sd,
          basis:'across 3 policies',notes:'Equal policy weights; differing eligible subsets. '+
            split.map(p=>p.policy+': '+p.model+' (≈'+p.value.toFixed(2)+'%; n='+p.n.toLocaleString('en-US')+')').join('; ')+
            '. Estimated from rounded ten-fold CV recall/specificity: 100 × (recall + specificity) / 2. '+
            'Not combined-policy performance, a confidence interval or fold SD. '+
            'Source: Stringham et al. (2021), Appendix S2.'})],
      hovertemplate:'%{customdata}<extra></extra>'});
    return {traces,layout};
  }
  async function mount(document,Plotly) {
    const host=document.querySelector('.summary-chart');
    if(!host) return;
    const summary=JSON.parse(document.getElementById('summary-data').textContent);
    const {traces,layout}=plotSpec(summary);
    host.style.height=layout.height+'px';
    const ready=Plotly.newPlot(host,traces,layout,
      {responsive:true,displayModeBar:false,scrollZoom:false});
    async function download(format) {
      await ready;
      const figure=document.createElement('div');
      figure.style.cssText='position:fixed;left:-10000px;top:0';document.body.append(figure);
      // Plotly mutates layout objects while rendering; start exports from a fresh spec.
      const {traces:exportTraces,layout:exported}=plotSpec(summary);
      const width=Math.max(1400,host.clientWidth);
      addExportGuide(exported,
        document.getElementById('summary-figure-title').textContent,
        document.getElementById('summary-guide'),'',width);
      const height=exported.height;
      try {
        await Plotly.newPlot(figure,exportTraces,exported,
          {staticPlot:true,displayModeBar:false,responsive:false});
        await Plotly.downloadImage(figure,{format,filename:'zero-shot-results-summary',
          width,height});
      } finally {Plotly.purge(figure);figure.remove();}
    }
    for(const format of ['png','svg']) {
      const button=document.querySelector('.summary-'+format);
      if(button) button.onclick=()=>download(format);
    }
    await ready;
  }
  const api={plotSpec,mount};
  if(typeof module!=='undefined'&&module.exports) module.exports=api;
  else {root.WildlifeSummary=api;mount(document,Plotly);}
})(typeof window!=='undefined'?window:globalThis);
