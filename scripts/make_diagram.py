import json
from pathlib import Path
els=[]
def base(t,x,y,w,h):
 i=len(els)+1
 return {'id':f'e{i}','type':t,'x':x,'y':y,'width':w,'height':h,'angle':0,'strokeColor':'#243a53','backgroundColor':'transparent','fillStyle':'solid','strokeWidth':2,'strokeStyle':'solid','roughness':1,'opacity':100,'groupIds':[],'frameId':None,'roundness':{'type':3} if t=='rectangle' else None,'seed':i*31,'version':1,'versionNonce':i*17,'isDeleted':False,'boundElements':None,'updated':1,'link':None,'locked':False}
def text(x,y,s,size=20):
 if '\n' in s:
  for j,line in enumerate(s.split('\n')):text(x,y+j*size*1.4,line,size)
  return
 o=base('text',x,y,max(map(len,s.split('\n')))*size*.58,len(s.split('\n'))*size*1.3)
 o.update({'text':s,'baseline':size,'fontSize':size,'fontFamily':1,'textAlign':'left','verticalAlign':'top','containerId':None,'originalText':s,'autoResize':True,'lineHeight':1.3});els.append(o)
def box(x,y,w,h,s,col):
 o=base('rectangle',x,y,w,h);o['backgroundColor']=col;els.append(o);text(x+20,y+20,s,18)
def arrow(x,y,w,h):
 o=base('arrow',x,y,w,h);o.update({'points':[[0,0],[w,h]],'lastCommittedPoint':None,'startBinding':None,'endBinding':None,'startArrowhead':None,'endArrowhead':'arrow','elbowed':False});els.append(o)
text(30,15,'Can a fusion setting learned on one dataset travel?',20)
box(30,85,300,130,'PILOT: SciFact\n300 labelled queries\nTune k and dense weight','#dbeafe')
arrow(340,148,65,0)
box(415,85,300,130,'FREEZE TWO SETTINGS\nBM25 + BGE-small\nBM25 + Qwen 0.6B','#fef3c7')
arrow(725,148,65,0)
box(800,85,330,130,'FRESH CONFIRMATION\nFiQA / NFCorpus / ArguAna\nFull corpora, all test queries','#dcfce7')
arrow(960,225,0,55)
box(30,295,300,145,'SINGLE RETRIEVERS\nBM25, BGE-small, Qwen\nChoose strongest baseline\nwithin each dataset','#dbeafe')
box(415,295,300,145,'TUNED HYBRIDS\nFuse top 1,000 ranks\nNo test-set adjustment\nQwen hybrid is primary','#fef3c7')
box(800,295,330,145,'PAIRED EVALUATION\nnDCG@10 + Recall@100\n10,000 bootstrap draws\nEqual weight per dataset','#dcfce7')
arrow(340,370,65,0);arrow(725,370,65,0)
arrow(960,450,0,55)
box(415,520,715,100,'DECISION: does the 95% interval clear +2 points?\nAbove margin: PROVEN | Below: DISPROVEN | Crosses: INCONCLUSIVE','#fce7f3')
text(30,540,'L4 GPU\nCheckpoint each batch\nPublic HF data only',18)
Path('figures/diagram.excalidraw').write_text(json.dumps({'type':'excalidraw','version':2,'source':'https://excalidraw.com','elements':els,'appState':{'viewBackgroundColor':'#ffffff','gridSize':None},'files':{}},indent=2))
