import json,hashlib
from pathlib import Path
R=Path('results');out={}
for ds in ['scifact','fiqa','nfcorpus','arguana']:
 p=R/ds/'data.json';d=json.loads(p.read_text())
 out[ds]={'revision':d['revision'],'qrels_revision':d['qrels_revision'],'documents':len(d['docids']),'queries':len(d['qids']),'data_json_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 # Small audit inputs permit score verification without storing all corpus text in Git.
 (R/ds/'evaluation_inputs.json').write_text(json.dumps({k:d[k] for k in ['docids','qids','qrels','queries']},indent=2))
(R/'data_manifest.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
