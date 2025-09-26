from pathlib import Path
import re
import pprint
import pandas as pd
from typing import Dict, List
import sbatchman as sbm
# from statistics import geometric_mean, stdev

OUT_DIR = Path('results')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_timer_string(timer_str):
  """Parse a timer string like 'n=2,avg=36.2,...' or '123.345' into a dict."""
  if ',' not in timer_str:
    return {'n': 1, 'avg': float(timer_str)}
  
  parts = timer_str.split(",")
  out = {}
  for p in parts:
    k, v = p.split("=")
    out[k.strip()] = float(v)
  return out

def runs_to_dataframe(runs):
  """
  Convert list of runs into a Pandas DataFrame.
  Each row = one (run, rank, timer_name).
  """
  records = []
  
  for run_id, run in enumerate(runs):
    for key, val in run.items():
      if key == "global_timer":
        # Global timer is a single entry
        rec = parse_timer_string(val)
        rec.update({
          "run": run_id,
          "rank": "global",
          "timer": "global_timer"
        })
        records.append(rec)
      else:
        # Key is rank (int)
        rank = key
        timers = val
        for timer_name, timer_str in timers.items():
          rec = parse_timer_string(timer_str)
          rec.update({
            "run": run_id,
            "rank": rank,
            "timer": timer_name
          })
          records.append(rec)
  
  return pd.DataFrame(records)

def sanitize_stdout_line(line: str) -> str:
  # Remove ANSI color codes at the start and end of the line
  line = re.sub(r'^\x1b\[[0-9;]*m', '', line)
  line = re.sub(r'\x1b\[0m$', '', line)
  return line

def parse_stdout(job: sbm.Job) -> List[Dict[str, Dict[str, str]]]:
  runs = []
  run_i = -1
  for line in job.get_stdout().splitlines():
    line = sanitize_stdout_line(line)
    
    # Parsing HnS stdout
    if 'hns' in job.tag.lower():
      if line.startswith('STARTING spgemm round'):
        run_i += 1
        runs.append({})
      elif line.startswith('<['):
        m = re.match(r'<\[process (\d+)\]>\[(\w+)\] (.+)', line)
        rank = int(m.group(1))
        timer_name = m.group(2)
        timer_data = m.group(3)
        if not runs[run_i].get(rank): runs[run_i][rank] = {}
        runs[run_i][rank][timer_name] = timer_data
      elif line.startswith('<Timer>[spgemm]'):
        runs[run_i]['global_timer'] = line.split(' ')[-1]
        
    # Parsing Trilinos stdout
    elif 'trilinos' in job.tag.lower():
      if line.startswith('<Timer>[spgemm] n='):
        runs[run_i]['global_timer'] = line.split(' ')[-1]
      elif line.startswith('<Timer>[spgemm]'):
        run_i += 1
        runs.append({})
        m = re.match(r'<Timer>\[(\w+)\] ([\d.]+) ms', line)
        timer_name = m.group(1)
        timer_data = m.group(2)
        rank = 'global'
        if not runs[run_i].get(rank): runs[run_i][rank] = {}
        runs[run_i][rank][timer_name] = timer_data

  return runs


def main():
  failed_jobs = sbm.jobs_list(status=[sbm.Status.FAILED], from_active=True, from_archived=True)
  if failed_jobs:
    print("WARNING: some jobs have failed!")
    pprint.pprint(failed_jobs)
    
  jobs = sbm.jobs_list(status=[sbm.Status.COMPLETED], from_active=True, from_archived=True)
  dfs = []

  for job in jobs:
    job_config: sbm.SlurmConfig = job.get_job_config()  # type: ignore
    job_config = sbm.SlurmConfig(**job_config.__dict__) if not isinstance(job_config, sbm.SlurmConfig) else job_config
    runs = parse_stdout(job)
    _, program, args = job.parse_command_args()
    if not program or not args:
      raise Exception(f'Could not parse program or args of job: {job}')
    if not job_config:
      raise Exception(f'Could not find config of job: {job}')
    
    print(job)
    pprint.pprint(runs)
    print('-'*50)
    
    mpi_async = False
    for env_v in (job_config.env if job_config.env else []):
      if env_v.lower().startswith('MPICH_ASYNC_PROGRESS') and env_v.split('=') == '1':
        mpi_async = True
        
    program = Path(program[0]).stem
    if 'trilinos' in program.lower():
      program = 'trilinos'
    elif 'run_spgemm' in program.lower():
      program = 'hns' # TODO add variant
      
    df = runs_to_dataframe(runs)
    df['cluster'] = job.cluster_name
    df['program'] = program
    df['nodes'] = job_config.nodes
    df['gpus'] = args['G']
    df['cpus_per_task'] = args['cpus-per-task']
    df['mpi_async'] = mpi_async
    df['grid'] = 'TODO' if 'hns' in program else '-'
    print(df)
    dfs.append(df)
    
  df = pd.concat(dfs, ignore_index=True)
  path = OUT_DIR / f'spgemm_{sbm.get_cluster_name()}_data.csv'
  df.to_csv(path, index=False)
  print(f'Data saved to {path.resolve().absolute()}')


if __name__ == "__main__":
  main()