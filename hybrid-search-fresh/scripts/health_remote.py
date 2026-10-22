from pathlib import Path
import subprocess,json
p=Path('/content/hybrid-search')
print(subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used','--format=csv'],text=True))
if (p/'worker.log').exists():print('\n'.join((p/'worker.log').read_text().splitlines()[-4:]))
for name in ['pilot_selection.json','worker_done.json','worker_error.json']:
 f=p/'results'/name
 if f.exists():print(name,f.read_text())
