import os,subprocess,tarfile,json
from pathlib import Path
p=Path('/content/hybrid-search')
print(subprocess.run(['tail','-12',str(p/'worker.log')],capture_output=True,text=True).stdout)
print('DONE', (p/'results/worker_done.json').exists(), 'ERROR',(p/'results/worker_error.json').exists())
print(subprocess.run(['ps','-eo','pid,etime,args'],capture_output=True,text=True).stdout[-2500:])
# Copy all textual artifacts and top rankings frequently. Embeddings remain on VM until final sync.
with tarfile.open('/content/hybrid-search/snapshot.tar.gz','w:gz') as tar:
 for f in (p/'results').rglob('*'):
  if f.is_file() and f.name != 'data.json' and f.suffix in ['.json','.csv','.npy']:tar.add(f,arcname=str(f.relative_to(p)))
 tar.add(p/'worker.log',arcname='figures/colab_run.txt')
