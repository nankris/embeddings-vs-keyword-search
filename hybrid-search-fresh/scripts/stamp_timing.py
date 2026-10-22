"""Track elapsed phases, preserving original pre-work estimate and start timestamp."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,time,sys
p=Path('results/timing.json');t=json.loads(p.read_text());now=time.time()
mode=sys.argv[1]
if mode in ['eval_start','eval_end']:t.setdefault(mode+'_unix',now)
elif mode=='finish':
 allocation=json.loads(Path('results/allocation.json').read_text());stop=json.loads(Path('results/stopped.json').read_text());start=datetime.fromisoformat(t['start_ist']).timestamp()
 t['end_ist']=datetime.fromtimestamp(now,timezone(timedelta(hours=5,minutes=30))).isoformat();t['actual_minutes']=(now-start)/60
 es=t['eval_start_unix'];ee=t['eval_end_unix']
 t['stage_minutes']={'setup':(allocation['unix']-start)/60,'GPU_run':(stop['unix']-allocation['unix'])/60,'eval':(ee-es)/60,'writeup':(now-ee)/60}
 t['stage_minutes']['setup']+=(es-stop['unix'])/60
 t['stage_accounting_note']='Elapsed phases, not exclusive active labor: setup includes initial setup, GPU-slot queue, interruption between attempts and post-stop sync. GPU phase includes detached inference, scoring, polling and concurrent report preparation. Eval is local audits/statistics/chart generation; writeup is final publication preparation.'
 t['GPU']='NVIDIA L4';t['colab_CU_estimated']=stop['estimated_cu']
else:raise ValueError(mode)
p.write_text(json.dumps(t,indent=2));print(json.dumps(t,indent=2))
