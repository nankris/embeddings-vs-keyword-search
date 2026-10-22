"""Short Colab polls, frequent local snapshots, budget guard and mandatory shutdown."""
import time,subprocess,json,tarfile,re
from pathlib import Path
R=Path('results');last_usage=0

def cli(args,timeout=100):
 p=subprocess.run(['colab']+args,capture_output=True,text=True,timeout=timeout)
 return p
while not (R/'launched.json').exists():time.sleep(10)
allocation=json.loads((R/'allocation.json').read_text());own_rate=1.54
while True:
 try:
  elapsed=time.time()-allocation['unix']
  if elapsed>3.5*3600:
   print('BUDGET GUARD STOP',flush=True);cli(['stop','-s','hybrid-search']);(R/'stopped.json').write_text(json.dumps({'reason':'budget_guard','unix':time.time(),'estimated_cu':elapsed*own_rate/3600}));break
  p=cli(['exec','-s','hybrid-search','-f','scripts/poll_remote.py']);(R/'latest_poll.txt').write_text(p.stdout+p.stderr)
  print(p.stdout[-5000:],flush=True)
  if p.returncode:time.sleep(30);continue
  d=cli(['download','-s','hybrid-search','/content/hybrid-search/snapshot.tar.gz','results/snapshot.tar.gz'])
  if d.returncode:print('DOWNLOAD failed',d.stdout,d.stderr,flush=True);time.sleep(30);continue
  with tarfile.open(R/'snapshot.tar.gz') as t:t.extractall('.',filter='data')
  if time.time()-last_usage>240:
   u=cli(['usage']);(R/'usage_checks.txt').open('a').write(f'{time.time()}\n'+u.stdout+u.stderr);last_usage=time.time();print(u.stdout,flush=True)
  if (R/'worker_done.json').exists() or (R/'worker_error.json').exists():
   reason='complete' if (R/'worker_done.json').exists() else 'worker_error'
   stop=cli(['stop','-s','hybrid-search']);print(stop.stdout+stop.stderr,flush=True)
   end=time.time();(R/'stopped.json').write_text(json.dumps({'reason':reason,'unix':end,'gpu_wall_minutes':(end-allocation['unix'])/60,'rate_cu_per_hour':own_rate,'estimated_cu':(end-allocation['unix'])*own_rate/3600,'stop_output':stop.stdout+stop.stderr},indent=2));break
 except Exception as e:print('POLL RETRY',repr(e),flush=True)
 time.sleep(45)
