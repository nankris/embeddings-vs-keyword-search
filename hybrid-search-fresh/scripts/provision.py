"""Wait for a free slot without touching another session; respect daily CU floor."""
import os,subprocess,time,json,re,tarfile
from pathlib import Path
from datetime import datetime,timezone
R=Path('results');state=R/'allocation_wait.json'
if state.exists():s=json.loads(state.read_text())
else:s={'started':time.time(),'next_attempt':time.time()+max(0,300-165),'attempts':0}
def persist():state.write_text(json.dumps(s,indent=2))
def cli(args):
 p=subprocess.run(['colab']+args,capture_output=True,text=True,timeout=180)
 print(datetime.now(timezone.utc).isoformat(),args,p.stdout,p.stderr,flush=True);return p
persist()
while time.time()-s['started']<7200:
 if time.time()<s['next_attempt']:time.sleep(min(20,s['next_attempt']-time.time()));continue
 sessions=cli(['sessions'])
 if '[hybrid-search]' in sessions.stdout:
  probe=cli(['exec','-s','hybrid-search','-f','scripts/resume_probe.py'])
  if '\"running\": true' in probe.stdout or '\"done\": true' in probe.stdout:
   (R/'launched.json').write_text(json.dumps({'unix':time.time(),'resumed':True}));break
  if '\"error\": true' in probe.stdout:raise RuntimeError('Existing worker failed; inspect error and stop named session')
 if '[hybrid-search]' not in sessions.stdout:
  usage=cli(['usage']);floor=float(Path(os.environ.get('CU_FLOOR_FILE','cu_floor.txt')).read_text().strip());match=re.search(r'Current balance:\s*([\d.]+)',usage.stdout)
  if not match:raise RuntimeError('Unable to parse balance, refusing allocation')
  if float(match[1])<=floor:s['next_attempt']=time.time()+900;persist();continue
  p=cli(['new','--gpu','L4','-s','hybrid-search']);s['attempts']+=1;s['next_attempt']=time.time()+300;persist()
  if p.returncode:continue
  allocation={'unix':time.time(),'utc':datetime.now(timezone.utc).isoformat(),'account_balance_before':float(match[1]),'floor':floor,'gpu':'L4'}
  (R/'allocation.json').write_text(json.dumps(allocation,indent=2))
 cli(['status','-s','hybrid-search'])
 with tarfile.open('results/input.tar.gz','w:gz') as tar:
  tar.add('scripts/run.py',arcname='run.py')
  for p in R.glob('*/*'):
   if p.is_file() and (p.name=='data.json' or p.name in ['bm25.npy','bm25_cost.json']):tar.add(p,arcname=str(p))
 up=cli(['upload','-s','hybrid-search','results/input.tar.gz','/content/hybrid_search_input.tar.gz'])
 if up.returncode:raise RuntimeError('Upload failed; named session needs resume')
 launch=cli(['exec','-s','hybrid-search','-f','scripts/launch_remote.py'])
 if launch.returncode:raise RuntimeError('Launch failed; inspect named session')
 (R/'launched.json').write_text(json.dumps({'unix':time.time(),'utc':datetime.now(timezone.utc).isoformat()}));break
else:raise RuntimeError('Two-hour allocation wait exhausted')
