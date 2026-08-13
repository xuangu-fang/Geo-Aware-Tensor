#!/usr/bin/env python3
from __future__ import annotations
import argparse,glob,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from geoaware.statistics import paired_seed_summary

def main():
 p=argparse.ArgumentParser();p.add_argument('inputs',nargs='+');p.add_argument('--output',required=True)
 p.add_argument('--proposed',default='tensor_tucker');p.add_argument('--prefix',default='tensor_t3');a=p.parse_args()
 files=[]
 for x in a.inputs:files += glob.glob(str(Path(x)/'seed_*.json')) if Path(x).is_dir() else glob.glob(x)
 vals=defaultdict(lambda:defaultdict(dict));params={}
 for f in sorted(set(files)):
  q=json.load(open(f));seed=int(q['config']['seed'])
  for key,v in q['summary'].items():
   split,model=key.split('/');
   if split!='unseen':continue
   params[model]=v['parameters']
   for metric in ('nrmse','high_band_nrmse','boundary_nrmse'):vals[metric][model][seed]=v[metric]
 baselines=['tensor_cp','tensor_tucker','diag_tucker','paired_cp','envelope_cp','envelope_tucker',
            'wrong_tensor','wrong_paired','wrong_envelope','wrong_envelope_tucker',
            'raw_finr_tucker','ipnf','neural_cp','siren']
 baselines=[b for b in baselines if b!=a.proposed and b in vals['nrmse']]
 paired={m:{b:paired_seed_summary(vals[m][a.proposed],vals[m][b]) for b in baselines} for m in vals}
 summary={m:{model:{'mean':float(np.mean(list(d.values()))),'std':float(np.std(list(d.values()),ddof=1)),'seeds':d,'parameters':params[model]} for model,d in models.items()} for m,models in vals.items()}
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/f'{a.prefix}_summary.json').write_text(json.dumps({'files':files,'proposed':a.proposed,'summary':summary,'paired':paired},indent=2))
 order=[a.proposed]+[x for x in ['envelope_cp','envelope_tucker','paired_cp','tensor_tucker','tensor_cp','diag_tucker',
                                 'wrong_envelope','wrong_envelope_tucker','wrong_tensor','wrong_paired','raw_finr_tucker','ipnf','neural_cp']
                     if x!=a.proposed and x in summary['nrmse']]
 labelmap={'envelope_cp':'Envelope CP','envelope_tucker':'Envelope Tucker','paired_cp':'Paired CP','tensor_tucker':'Geo Tucker',
           'tensor_cp':'Geo CP','diag_tucker':'Diag core','wrong_tensor':'Wrong geom.',
           'wrong_paired':'Wrong paired','wrong_envelope':'Wrong envelope','wrong_envelope_tucker':'Wrong env. Tucker',
           'raw_finr_tucker':'Raw F-INR','ipnf':'IP-NF','neural_cp':'Neural CP'}
 labels=[labelmap[x] for x in order]
 fig,ax=plt.subplots(figsize=(8,3.8));means=[summary['nrmse'][x]['mean'] for x in order];std=[summary['nrmse'][x]['std'] for x in order]
 ax.bar(range(len(order)),means,yerr=std,capsize=3,color=['#176b87']+['#3c91a8' if 'tensor' in x or x=='paired_cp' else '#d28e42' for x in order[1:]])
 ax.set_ylabel('Unseen 24→32 NRMSE');ax.set_xticks(range(len(order)),labels,rotation=25,ha='right');ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(out/f'{a.prefix}_crossres.png',dpi=180)
 print(json.dumps(paired['nrmse'],indent=2))
if __name__=='__main__':main()
