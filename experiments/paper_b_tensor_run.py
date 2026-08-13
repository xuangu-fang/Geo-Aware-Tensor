#!/usr/bin/env python3
"""Explicit tensor-factor Paper-B iteration runner."""
from __future__ import annotations
import argparse,json,time,random
from pathlib import Path
import numpy as np, torch
from geoaware.neural_geometry import CoordinateField, IntrinsicPhaseField, SharedNeuralCP, make_tasks, task_coordinate_features
from geoaware.neural_tensor import (GeometryNeuralCP,GeometryNeuralTucker,
                                    SpeedAlignedPhaseCP,PhaseEnvelopeCP,
                                    PhaseEnvelopeTucker)
from paper_b_run import build_family,metrics

def seed_all(s): random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)

def fit(model,tasks,name,steps,lr,device):
    model.to(device); model.train();opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-6)
    ys=torch.cat([t.noisy_values[t.observed] for t in tasks]).to(device);scale=ys.std().clamp_min(1e-6)
    tensor_names=('tensor_cp','tensor_tucker','wrong_tensor','no_phase_tensor','diag_tucker',
                  'raw_finr_tucker','paired_cp','wrong_paired','envelope_cp','wrong_envelope',
                  'envelope_tucker','wrong_envelope_tucker')
    if name in tensor_names+('ipnf',):
        feats=[model.feature_task(t,name in ('wrong_tensor','raw_finr_tucker','wrong_paired',
                                             'wrong_envelope','wrong_envelope_tucker')) for t in tasks]
        if name!='ipnf':
            gs=[];ts=[];xs=[]
            for task,(g,t,x) in zip(tasks,feats):
                n=int(task.observed.sum());gs.append(g[None].expand(n,-1));ts.append(t[None].expand(n,-1));xs.append(x[task.observed])
            point_features=(torch.cat(gs).to(device),torch.cat(ts).to(device),torch.cat(xs).to(device))
    else: xs=torch.cat([task_coordinate_features(t)[t.observed] for t in tasks]).to(device)
    hist=[]
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        if name in tensor_names:
            pred=model.forward_points(*point_features)
        elif name=='ipnf':
            ps=[]
            for t,f in zip(tasks,feats):ps.append(model(f.to(device)[t.observed]))
            pred=torch.cat(ps)
        else: pred=model(xs)
        extra=model.regularization() if hasattr(model,'regularization') else pred.new_zeros(())
        loss=((pred/scale)-(ys/scale)).square().mean()+1e-7*sum(p.square().mean() for p in model.parameters())+2e-4*extra
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if step%200==0 or step==steps-1:hist.append({'step':step,'loss':float(loss.detach())})
    return hist

@torch.no_grad()
def predict(model,t,name,device):
    model.eval()
    if name in ('tensor_cp','tensor_tucker','wrong_tensor','no_phase_tensor','diag_tucker',
                'raw_finr_tucker','paired_cp','wrong_paired','envelope_cp','wrong_envelope',
                'envelope_tucker','wrong_envelope_tucker'):
        return model.forward_task(t,wrong_geometry=name in ('wrong_tensor','raw_finr_tucker',
                                  'wrong_paired','wrong_envelope','wrong_envelope_tucker')).cpu()
    if name=='ipnf': return model(model.feature_task(t,False).to(device)).cpu()
    return model(task_coordinate_features(t).to(device)).cpu()

def run(a):
    seed_all(a.seed);dev=torch.device(a.device if torch.cuda.is_available() else 'cpu')
    tr,te=build_family(a.train_resolution,a.n_eigen,a.geometry_family)
    if a.test_resolution!=a.train_resolution:
        from paper_b_run import NARROW_TEST_SPECS
        from geoaware.neural_geometry import build_obstacle_domain
        te=[build_obstacle_domain(x,a.test_resolution,a.n_eigen) for x in NARROW_TEST_SPECS]
    train=make_tasks(tr,a.train_times,a.ratio,a.seed,a.noise,'random',a.target_kind,a.mismatch)
    test=make_tasks(te,a.test_times,a.ratio,a.seed+999,a.noise,'random',a.target_kind,a.mismatch)
    in_dim=task_coordinate_features(train[0]).shape[1]
    factories={
      'tensor_cp':lambda:GeometryNeuralCP(a.rank,a.hidden,True),
      'tensor_tucker':lambda:GeometryNeuralTucker((a.geo_rank,a.time_rank,a.space_rank),a.hidden,True,band_gates=a.band_gates),
      'wrong_tensor':lambda:GeometryNeuralCP(a.rank,a.hidden,True),
      'no_phase_tensor':lambda:GeometryNeuralCP(a.rank,a.hidden,False),
      'diag_tucker':lambda:GeometryNeuralTucker((a.geo_rank,a.time_rank,a.space_rank),a.hidden,True,True),
      'raw_finr_tucker':lambda:GeometryNeuralTucker((a.geo_rank,a.time_rank,a.space_rank),a.hidden,False),
      'paired_cp':lambda:SpeedAlignedPhaseCP(max(32,a.hidden*3//4)),
      'wrong_paired':lambda:SpeedAlignedPhaseCP(max(32,a.hidden*3//4)),
      'envelope_cp':lambda:PhaseEnvelopeCP(a.envelope_rank,max(32,a.hidden*3//4)),
      'wrong_envelope':lambda:PhaseEnvelopeCP(a.envelope_rank,max(32,a.hidden*3//4)),
      'envelope_tucker':lambda:PhaseEnvelopeTucker(a.envelope_distance_rank,a.envelope_time_rank,max(32,a.hidden*3//4)),
      'wrong_envelope_tucker':lambda:PhaseEnvelopeTucker(a.envelope_distance_rank,a.envelope_time_rank,max(32,a.hidden*3//4)),
      'ipnf':lambda:IntrinsicPhaseField(in_dim,a.hidden,True),
      'neural_cp':lambda:SharedNeuralCP(in_dim-2,a.rank,max(32,a.hidden//2)),
      'siren':lambda:CoordinateField(in_dim,a.hidden,'siren',a.seed),
    }
    requested=set(a.models.split(',')) if a.models else set(factories)
    models={}
    for name,factory in factories.items():
        if name in requested:
            # Model construction is independently seeded.  Thus adding or
            # removing an unrelated baseline cannot change another model's
            # initialization, and correct/wrong controls start identically.
            seed_all(a.seed)
            models[name]=factory()
    rows=[];history={};started=time.time()
    for name,model in models.items():
        seed_all(a.seed);history[name]=fit(model,train,name,a.steps,a.lr,dev)
        for split,tasks in [('seen',train),('unseen',test)]:
            for t in tasks:
                rows.append({'seed':a.seed,'model':name,'split':split,'task':t.name,
                             'parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                             **metrics(t,predict(model,t,name,dev))})
    summary={}
    for split in ('seen','unseen'):
      for name in models:
       rr=[r for r in rows if r['split']==split and r['model']==name]
       summary[f'{split}/{name}']={k:float(np.mean([r[k] for r in rr])) for k in ('nrmse','high_band_nrmse','boundary_nrmse','shadow_nrmse')}
       summary[f'{split}/{name}']['parameters']=rr[0]['parameters']
       if hasattr(models[name],'band_summary'):summary[f'{split}/{name}']['band_gates']=models[name].band_summary()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    payload={'config':vars(a),'elapsed_seconds':time.time()-started,'summary':summary,'rows':rows,'history':history}
    (out/f'seed_{a.seed}.json').write_text(json.dumps(payload,indent=2));print(json.dumps(summary,indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument('--round',required=True);p.add_argument('--seed',type=int,default=0)
 p.add_argument('--ratio',type=float,default=.02);p.add_argument('--noise',type=float,default=.05)
 p.add_argument('--train-resolution',type=int,default=24);p.add_argument('--test-resolution',type=int,default=24);p.add_argument('--n-eigen',type=int,default=96)
 p.add_argument('--rank',type=int,default=32);p.add_argument('--geo-rank',type=int,default=6);p.add_argument('--time-rank',type=int,default=10);p.add_argument('--space-rank',type=int,default=16)
 p.add_argument('--hidden',type=int,default=64);p.add_argument('--steps',type=int,default=900);p.add_argument('--lr',type=float,default=.002);p.add_argument('--device',default='cuda')
 p.add_argument('--envelope-rank',type=int,default=4)
 p.add_argument('--envelope-distance-rank',type=int,default=10);p.add_argument('--envelope-time-rank',type=int,default=6)
 p.add_argument('--band-gates',action='store_true')
 p.add_argument('--geometry-family',default='narrow_wall');p.add_argument('--train-times',nargs='+',type=float,default=[.16,.24,.32,.40]);p.add_argument('--test-times',nargs='+',type=float,default=[.20,.28,.36]);p.add_argument('--output',required=True)
 p.add_argument('--target-kind',choices=['geodesic','harmonic','mixed'],default='geodesic')
 p.add_argument('--mismatch',type=float,default=0.)
 p.add_argument('--models',default='')
 run(p.parse_args())
if __name__=='__main__':main()
