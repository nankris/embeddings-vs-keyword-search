import subprocess,os,tarfile
os.makedirs('/content/hybrid-search',exist_ok=True)
with tarfile.open('/content/hybrid_search_input.tar.gz') as tar:tar.extractall('/content/hybrid-search',filter='data')
cmd="cd /content/hybrid-search && python -m pip install -q 'sentence-transformers>=5,<6' rank_bm25 pyarrow pandas && OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -u run.py"
p=subprocess.Popen(['nohup','bash','-c',cmd],stdout=open('/content/hybrid-search/worker.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
print('DETACHED_PID',p.pid)
open('/content/hybrid-search/pid','w').write(str(p.pid))
