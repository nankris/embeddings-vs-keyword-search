import subprocess,os,tarfile
os.makedirs('/content/exp120b',exist_ok=True)
with tarfile.open('/content/120b_input.tar.gz') as tar:tar.extractall('/content/exp120b',filter='data')
cmd="cd /content/exp120b && python -m pip install -q 'sentence-transformers>=5,<6' rank_bm25 pyarrow pandas && OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -u run.py"
p=subprocess.Popen(['nohup','bash','-c',cmd],stdout=open('/content/exp120b/worker.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
print('DETACHED_PID',p.pid)
open('/content/exp120b/pid','w').write(str(p.pid))
