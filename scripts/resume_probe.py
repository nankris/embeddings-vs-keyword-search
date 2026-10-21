from pathlib import Path
import os,json
p=Path('/content/exp120b');running=False
if (p/'pid').exists():
 try:os.kill(int((p/'pid').read_text()),0);running=True
 except (ProcessLookupError,ValueError):pass
print(json.dumps({'running':running,'done':(p/'results/worker_done.json').exists(),'error':(p/'results/worker_error.json').exists()}))
if (p/'worker.log').exists():print((p/'worker.log').read_text()[-1500:])
