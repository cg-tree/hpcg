import os
import sys
from datetime import timedelta

'''mfricke@easley:~HPCG/$
module module load gcc/14.2.0-cuda-jeua apptainer
mfricke@easley:~HPCG/$
srun –mem 32GB --mpi=pmi2 --
partition l40s --nodes 1 --ntasks-per-node=1 --gpus 1
singularity run --nv --bind .:/my-dat-files hpc-benchmarks\:24.03.sif ./hpcg.sh --dat hpcg.dat'''


header = '''#!/bin/bash
#SBATCH --partition debug
#SBATCH --nodes 1

'''
setup = '''
mkdir $SLURM_JOB_ID
cp delivery $SLURM_SUBMIT_DIR/$SLURM_JOB_ID/

export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores

'''
workload = '''
srun –mem 32GB --mpi=pmi2 --
partition l40s --nodes 1 --ntasks-per-node=1 --gpus 1
singularity run --nv --bind .:$SLURM_SUBMIT_DIR hpc-benchmarks\:24.03.sif ./hpcg.sh --dat hpcg.dat
'''

postprocess = '''
mv slurm-$SLURM_JOB_ID.out $SLURM_SUBMIT_DIR/$SLURM_JOB_ID
python3 postprocess.py
'''

fname = 'hpcg'
Ns = [10,100,1000,10000]
mems = [4,8,16,32,64]
times = ['00:00:01','00:00:01','00:00:01','00:00:30','00:01:40']

nprocs = 32

sockets_per_node = [1,2]

#defaults
nindex = 0
nprocs = 32
nsockets = 1

slurm_vars = ["SLURM_ARRAY_TASK_COUNT", "SLURM_ARRAY_TASK_ID",
              "SLURM_ARRAY_TASK_MAX", "SLURM_ARRAY_TASK_MIN",
              "SLURM_ARRAY_TASK_STEP", "SLURM_ARRAY_JOB_ID",
              "SLURM_CLUSTER_NAME", "SLURM_CPUS_ON_NODE",
              "SLURM_CPUS_PER_TASK", "SLURM_JOB_ACCOUNT",
              "SLURM_JOBID,", "SLURM_JOB_ID",
              "SLURM_JOB_CPUS_PER_NODE", "SLURM_JOB_DEPENDENCY",
              "SLURM_JOB_NAME", "SLURM_NODELIST,",
              "SLURM_JOB_NODELIST", "SLURM_NNODES,",
              "SLURM_JOB_NUM_NODES", "SLURM_MEM_PER_NODE",
              "SLURM_MEM_PER_CPU", "SLURM_NTASKS,", "SLURM_NPROCS",
              "SLURM_NTASKS_PER_NODE", "SLURM_NTASKS_PER_SOCKET",
              "SLURM_SUBMIT_DIR", "SLURM_SUBMIT_HOST",
              "SLURM_TASK_PID", "SLURMD_NODENAME", "SLURM_JOB_GPUS"]

slurm_args = ["--job-name",
    "--partition",
    "--nodes",
    "--ntasks-per-node",
    "--mem",
    "--output",
    "--error"]

def get_slurm_env():
  env = dict()
  vals = [ os.getenv(var) for var in slurm_vars]
  for i in range(len(slurm_vars)):
    env[slurm_vars[i]] = vals[i]
  return env

def write_slurm_script(filename,
                       header = header,
                       setup = setup,
                       workload = workload,
                       postprocess = postprocess):

  with open(filename,'w') as f:
    f.writelines(header)
    f.writelines(setup)
    f.writelines(workload)
    f.writelines(postprocess)


def sweep(start,count,inc):
  return [x*start + (x-1)*inc for x in range(count)]

def sbatch_s(val, index = 1, arg = ""):
  if arg == "":
    arg = slurm_args[index]
  return f"#SBATCH {arg} {val}"

def mod_header(nindex, ntasks, nthreads, nsockets):
  threadspersock = int(nthreads / nsockets)
  taskpersock = int(ntasks / nsockets)
  tseconds = int(Ns[nindex]/ (nthreads * 2))
  
  cpus = sbatch_s( int(nthreads/ntasks), arg = '--cpus-per-task')
  time = sbatch_s( str(timedelta(seconds = tseconds)), arg="--time")
  mem = sbatch_s(mems[nindex], arg="--mem")+"GB"
  tasks = sbatch_s(nthreads, arg='--ntasks-per-node')
  sockettasks = sbatch_s(taskpersock, arg='--ntasks-per-socket')
  sockets = sbatch_s(nsockets, arg='--sockets-per-node')
 
  headermods = [header, time, mem, tasks, sockettasks, sockets]
  return '\n'.join(headermods)


def mod_setup(nprocs, fname=fname):
  setup = ['\n',
          'mkdir $SLURM_JOB_ID',
          f"mv {fname} $SLURM_SUBMIT_DIR/$SLURM_JOB_ID/",
          'cp hpcg.dat $SLURM_SUBMIT_DIR/$SLURM_JOB_ID/hpcg.dat',
          "module load gcc/14.2.0-cuda-jeua apptainer",
          "export OMP_PROC_BIND=TRUE",
          "export OMP_PLACES=cores",
          '\n']
  # the number of threads spawned by each mpi process 
  omp_threads = f"export OMP_NUM_THREADS={nprocs}"
  setupmods = ['\n'] + setup + [omp_threads]

  return '\n'.join(setupmods)

def mod_workload(launcher = 'srun --mpi=pmi2',
                 hostdir = '.',
                 datadir = '$SLURM_SUBMIT_DIR',
                 inputfile = 'hpcg.dat'):

  return f'\n{launcher} singularity run --nv --bind {hostdir}:{datadir} hpc-benchmarks\:25.09.sif ./hpcg.sh --dat {inputfile}'

def mod_postprocess():
  postprocess = 'mv slurm-$SLURM_JOB_ID.out $SLURM_SUBMIT_DIR/$SLURM_JOB_ID'
  return '\n'.join(['\n']+[postprocess, '\n'])



def write_submit_sweep(count,sweepname ="submit_sweep",filename = fname):
  lines = [ f"sbatch {filename}-{i}" for i in range(count)]
  with open(sweepname, 'w') as f:
    f.writelines('\n'.join(lines) )
      


def write_ahmdal_gpu_sweep():
  nodes = [1,2,3,4]
  gpus = [1,2]
  procs = [1,2,4]
  index = 0
  for nnodes in nodes:
    for ngpus in gpus:
      for nprocs in procs:
        filename = '-'.join( ['ahmdal_gpu',str(index)] )
        header= ["#!/bin/bash",
                sbatch_s('h100', arg='--partition'),
                sbatch_s(nnodes, arg='--nodes'),
                sbatch_s(ngpus, arg = '--gpus-per-node'),
                sbatch_s(nprocs, arg = '--ntasks-per-node'),
                sbatch_s(max(1,int(nprocs/2)), arg='--ntasks-per-socket'),
                sbatch_s(min(2,nprocs), arg='--sockets-per-node'),
                sbatch_s('32GB', arg = '--mem'),
                sbatch_s('00:01:30', arg = '--time')]

  
        setup = ['\n',
          'mkdir $SLURM_JOB_ID',
          f"mv {filename} $SLURM_SUBMIT_DIR/$SLURM_JOB_ID/",
          "CONT='hpc-benchmarks:25.09.sif'",
          'MOUNT=".:/my-dat-files"']
        N = nnodes * nprocs
        workload = ['srun --mpi=pmi2 singularity exec --nv --bind',
                "${SLURM_SUBMIT_DIR}:/my-dat-files",
                "$CONT" ,
                'bash -c "cd /workspace && ./hpcg.sh --nx 128 --ny 128 --nz 128 --rt 60 --mem-affinity 0:0:1:1"']

        #workload = [f'srun -N {N} --mpi=pmix',
        #        '--container-image="${CONT}"',
        #         '--container-mounts="${MOUNT}"',
        #         './hpcg.sh --nx 512 --ny 512 --nz 256 --rt 2 --mem-affinity 0:0:1:1']
        write_slurm_script(filename,
                     header='\n'.join( header+ ['\n']),
                     setup='\n'.join(['\n'] + setup + ['\n']),
                     workload=' '.join(workload),
                     postprocess = mod_postprocess())

        index +=1
  write_submit_sweep(index,'submit_ahmdal_gpu',filename='ahmdal_gpu')




'''gustafson (nx_1*ny_1*nz_1)/p_1 = (nx_2*ny_2*nz_2)/p_2
constant work per process
let p_2 = 2*p_1
'''
def gustafson_next_size(nx,ny,nz):
  mn = min(nx, ny, nz)
  if nx == mn:
    nx*=2
  elif ny == mn:
    ny*=2
  else:
    nz*=2
  return (nx, ny, nz)


def write_gustafson_gpu_sweep():
  nodes = [1,2]
  gpus = [1,2]
  procs = [1,2]
  index = 0
  n = (512,16,16)
  for nnodes in nodes:
    for ngpus in gpus:
      for nprocs in procs:
        filename = '-'.join( ['gustafson_gpu',str(index)] )
        ntasks = ngpus * nprocs
        nsockets = min(2,ntasks)
        header= ["#!/bin/bash",
                sbatch_s('h100', arg='--partition'),
                sbatch_s(nnodes, arg='--nodes'),
                sbatch_s(ngpus, arg = '--gpus-per-node'),
                sbatch_s(nprocs, arg = '--cpus-per-gpu'),
                sbatch_s(int(ntasks/nsockets), arg='--ntasks-per-socket'),
                sbatch_s(nsockets, arg='--sockets-per-node'),
                sbatch_s('32GB', arg = '--mem'),
                sbatch_s('00:01:30', arg = '--time')]

  
        setup = ['\n',
          'mkdir $SLURM_JOB_ID',
          f"mv {filename} $SLURM_SUBMIT_DIR/$SLURM_JOB_ID/",
          "CONT='hpc-benchmarks:25.09.sif'",
          'MOUNT=".:/my-dat-files"']
        N = nnodes * nprocs
        workload = ['srun --mpi=pmi2 singularity exec --nv --bind',
                "${SLURM_SUBMIT_DIR}:/my-dat-files",
                "$CONT" ,
                f'bash -c "cd /workspace && ./hpcg.sh --nx {n[0]} --ny {n[1]} --nz {n[2]} --rt 60 --mem-affinity 0:0:1:1"']

        write_slurm_script(filename,
                     header='\n'.join( header+ ['\n']),
                     setup='\n'.join(['\n'] + setup + ['\n']),
                     workload=' '.join(workload),
                     postprocess = mod_postprocess())

        n = gustafson_next_size(n[0],n[1],n[2])
        index +=1
  write_submit_sweep(index,'submit_gustafson_gpu',filename='gustafson_gpu')

"""
write a bunch of related sbatch scripts
"""
def write_parameter_sweep():
  
  process_counts = [2<<x for x in reversed(range(4))]
  index = 0
  for nindex in range(len(process_counts)):
    for nprocs in process_counts:
      for sockets in sockets_per_node:
        filename = '-'.join( [fname,str(index)] )

        write_slurm_script(filename,
                     header=mod_header(nindex,nprocs,1,sockets),
                     setup=mod_setup( nprocs, filename),
                     workload=mod_workload(),
                     postprocess = mod_postprocess())
        index += 1
  return index

if __name__ == '__main__':
  write_submit_sweep( write_gustafson_gpu_sweep(),'submit_gustafson_gpu', filename='gustafson_gpu' )

