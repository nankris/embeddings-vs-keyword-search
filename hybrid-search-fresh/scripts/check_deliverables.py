"""Validate the deliverable bundle without rerunning model inference."""
import json,re
from pathlib import Path
from PIL import Image
root=Path('.')
required=['README.md','writeup.md','blog.md','results/analysis.json','results/audit.json','results/timing.json','results/stopped.json','figures/diagram.excalidraw','figures/diagram.png','figures/table.png','figures/scores.png','figures/gains.png','figures/recall.png','figures/colab_run.txt']
for x in required:assert (root/x).is_file(),x
blog=Path('blog.md').read_text();words=len(re.findall(r"\b[\w'-]+\b",blog));assert 1200<=words<=1700,words
assert '\u2014' not in blog
assert 'Code:' in blog and 'https://' in blog
for x in ['scores','gains','recall','table']:
 im=Image.open(f'figures/{x}.png');assert min(im.size)>400,(x,im.size)
 assert all(abs(v-150)<1 for v in im.info['dpi']),im.info
assert json.loads(Path('figures/diagram.excalidraw').read_text())['type']=='excalidraw'
a=json.loads(Path('results/analysis.json').read_text());lo,hi=a['primary']['ci95'];assert a['verdict']==('PROVEN' if lo>=.02 else 'DISPROVEN' if hi<.02 else 'INCONCLUSIVE')
s=json.loads(Path('results/stopped.json').read_text());assert s['estimated_cu']<6 and s['reason']=='complete'
t=json.loads(Path('results/timing.json').read_text());assert t['actual_minutes']>0 and 'end_ist' in t and all(k in t['stage_minutes'] for k in ['setup','GPU_run','eval','writeup'])
assert 'NVIDIA L4' in Path('figures/colab_run.txt').read_text() and 'WORKER COMPLETE' in Path('figures/colab_run.txt').read_text()
print(f'PASS: publication bundle, {words} blog words, PNG resolution, fixed verdict rule, timing, CU cap and GPU evidence')
