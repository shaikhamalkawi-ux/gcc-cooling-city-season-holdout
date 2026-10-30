#!/usr/bin/env python3
"""Run only R10 numerical source-to-result replay, hash-compare each canonical output."""
from pathlib import Path
import hashlib,json,sys,tempfile,subprocess
root=Path(__file__).resolve().parents[1]
source=root/'source'/'Project82_R04_Frozen_2023_Source.zip'
script=root/'code'/'r10_double_block_validation.py'
expected=root/'results'
expected_files=[x.name for x in expected.iterdir() if x.name.startswith('R10_') and x.suffix=='.csv']+['R10_Strict_SplitProtocol_And_Gates.json']
def shaf(p):return hashlib.sha256(p.read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(prefix='Project82_R10_replay_') as tmp:
 ret=subprocess.run([sys.executable,str(script),'--source',str(source),'--output',tmp],capture_output=True,text=True,timeout=90)
 if ret.returncode:
  raise RuntimeError('Source replay failed: '+ret.stderr[-1200:])
 issues=[];verified=[]
 for name in sorted(expected_files):
  a=expected/name;b=Path(tmp)/name
  if not b.exists() or shaf(a)!=shaf(b):issues.append(name)
  else:verified.append({'file':name,'sha256':shaf(a)})
 if issues:raise RuntimeError('Failed byte-identical outputs: '+repr(issues))
 report={'status':'PASS','source_sha256':shaf(source),'source_file':source.name,'byte_identical_files':len(verified),
 'verified_files':verified,'scope':'calculation outputs only, not fresh EnergyPlus run','archive_gates':'R05=HOLD; no new five-city simulation'}
 out=root/'verification'/'R10_FreshReplay_ByteIdentity.json';out.write_text(json.dumps(report,indent=2))
 print(f'PASS: {len(verified)} scientific output files byte-identical in fresh isolated replay')
