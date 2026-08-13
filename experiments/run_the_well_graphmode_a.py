#!/usr/bin/env python3
"""Full-2D graph-mode Bayesian Tucker smoke for Paper A."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch import nn

from geoaware.the_well_pilot import fixed_random_mask, load_the_well_case


def graph_eigenbasis(speed: np.ndarray, modes: int, flat: bool = False):
    nx, ny = speed.shape
    fluid = np.ones_like(speed, dtype=bool) if flat else speed > .1
    ids = np.full((nx, ny), -1, dtype=np.int64)
    ids[fluid] = np.arange(int(fluid.sum()))
    conductance = np.ones_like(speed, dtype=np.float64) if flat else speed.astype(np.float64)**2
    edges_i, edges_j, weights = [], [], []
    for left, right, c_left, c_right in (
            (ids[:-1], ids[1:], conductance[:-1], conductance[1:]),
            (ids[:, :-1], ids[:, 1:], conductance[:, :-1], conductance[:, 1:])):
        valid = (left >= 0) & (right >= 0)
        a, b = left[valid], right[valid]
        w = 2/(1/c_left[valid]+1/c_right[valid])
        edges_i.extend(a); edges_j.extend(b); weights.extend(w)
    a, b, w = np.asarray(edges_i), np.asarray(edges_j), np.asarray(weights)
    laplacian = sp.coo_matrix((np.r_[w, w, -w, -w],
                               (np.r_[a, b, a, b], np.r_[a, b, b, a])),
                              shape=(int(fluid.sum()), int(fluid.sum()))).tocsr()
    # The six-order material contrast makes the graph nearly disconnected and
    # plain ``which=SM`` ARPACK can fail after tens of thousands of iterations.
    # A small negative shift is outside the PSD spectrum and robustly targets
    # the same lowest modes without regularizing away the maze topology.
    values, vectors = spla.eigsh(laplacian, k=modes, sigma=-1e-5,
                                 which="LM", tol=1e-6)
    order = np.argsort(values)
    values, vectors = values[order], vectors[:, order]
    residual = np.linalg.norm(laplacian@vectors-vectors*values[None], axis=0)
    embedded = np.zeros((nx*ny, modes), dtype=np.float32)
    embedded[np.flatnonzero(fluid)] = vectors.astype(np.float32)
    return (torch.from_numpy(embedded),
            torch.from_numpy(values.astype(np.float32)), laplacian,
            float(residual.max()))


def cosine_time_basis(length: int, modes: int):
    grid = torch.arange(length).float()[:, None]
    frequency = torch.arange(modes).float()[None]
    basis = torch.cos(math.pi*(grid+.5)*frequency/length)
    basis[:, 0] /= math.sqrt(2.)
    basis *= math.sqrt(2./length)
    eigenvalues = 4*torch.sin(math.pi*torch.arange(modes).float()/(2*length)).square()
    return basis, eigenvalues


class GraphModeFactorization(nn.Module):
    def __init__(self, graph_bases: torch.Tensor, time_basis: torch.Tensor,
                 ranks=(6, 10, 10), cp_rank: int | None = None):
        super().__init__()
        self.register_buffer("graph_bases", graph_bases)
        self.register_buffer("time_basis", time_basis)
        self.cp_rank = cp_rank
        g, _, spatial_modes = graph_bases.shape
        time_modes = time_basis.shape[1]
        if cp_rank is None:
            rg, rt, rs = ranks
            self.instance = nn.Parameter(torch.randn(g, rg)/math.sqrt(g))
            self.time_coeff = nn.Parameter(torch.randn(time_modes, rt)/math.sqrt(time_modes))
            self.space_coeff = nn.Parameter(torch.randn(spatial_modes, rs)/math.sqrt(spatial_modes))
            self.core = nn.Parameter(torch.randn(rg, rt, rs)/math.sqrt(rg*rt*rs))
        else:
            self.instance = nn.Parameter(torch.randn(g, cp_rank)/math.sqrt(g))
            self.time_coeff = nn.Parameter(torch.randn(time_modes, cp_rank)/math.sqrt(time_modes))
            self.space_coeff = nn.Parameter(torch.randn(spatial_modes, cp_rank)/math.sqrt(spatial_modes))
            self.core = nn.Parameter(torch.ones(cp_rank)/math.sqrt(cp_rank))
        self.posterior = None

    def factors(self, indices: torch.Tensor):
        g, t, node = indices.T
        instance = self.instance[g]
        temporal = self.time_basis[t]@self.time_coeff
        spatial = torch.einsum("nk,nkr->nr", self.graph_bases[g, node],
                               self.space_coeff[None].expand(len(indices), -1, -1))
        return instance, temporal, spatial

    def design(self, indices: torch.Tensor):
        instance, temporal, spatial = self.factors(indices)
        if self.cp_rank is not None:
            return instance*temporal*spatial
        return torch.einsum("na,nb,nc->nabc", instance, temporal, spatial).flatten(1)

    def forward(self, indices: torch.Tensor):
        return self.design(indices)@self.core.flatten()

    def regularization(self, time_eigenvalues, graph_eigenvalues):
        time_precision = (1+time_eigenvalues[:, None]).pow(1.5)
        graph_precision = (1+graph_eigenvalues.mean(0)[:, None]).pow(1.5)
        return (self.instance.square().mean() +
                (time_precision*self.time_coeff.square()).mean() +
                (graph_precision*self.space_coeff.square()).mean() +
                self.core.square().mean())

    def fit(self, indices, values, time_eigenvalues, graph_eigenvalues,
            steps=500, seed=0, batch_size=8192):
        device = self.core.device
        indices, values = indices.to(device), values.to(device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-3, weight_decay=1e-6)
        generator = torch.Generator(device=device).manual_seed(seed)
        best = (float("inf"), None)
        for _ in range(steps):
            chosen = torch.randint(len(indices), (min(batch_size, len(indices)),),
                                   generator=generator, device=device)
            prediction = self(indices[chosen])
            loss = (prediction-values[chosen]).square().mean()+2e-3*self.regularization(
                time_eigenvalues.to(device), graph_eigenvalues.to(device))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.); optimizer.step()
            score = float(loss.detach())
            if score < best[0]:
                best = (score, {key: value.detach().cpu().clone()
                                for key, value in self.state_dict().items()})
        self.load_state_dict(best[1]); self.to(device)
        with torch.no_grad():
            design = self.design(indices).double(); y = values.double()
            p = design.shape[1]; eye = torch.eye(p, device=device, dtype=torch.float64)
            alpha = torch.tensor(1., device=device, dtype=torch.float64)
            beta = torch.tensor(25., device=device, dtype=torch.float64)
            for _ in range(40):
                covariance = torch.linalg.inv(beta*(design.T@design)+alpha*eye)
                mean = beta*(covariance@design.T@y)
                gamma = (p-alpha*covariance.trace()).clamp(1e-3, p-1e-3)
                alpha = (gamma/mean.square().sum().clamp_min(1e-8)).clamp(1e-4, 1e5)
                residual = (y-design@mean).square().sum()
                beta = ((len(y)-gamma).clamp_min(1.)/residual.clamp_min(1e-8)).clamp(1e-3, 1e5)
            covariance = torch.linalg.inv(beta*(design.T@design)+alpha*eye)
            mean = beta*(covariance@design.T@y)
            self.posterior = {"mean": mean.float(), "covariance": covariance.float(),
                              "noise": float(beta.rsqrt()), "alpha": float(alpha)}
        return self

    @torch.no_grad()
    def predict(self, indices, chunk=32768):
        device = self.core.device; means=[]; variances=[]
        mean_core = self.posterior["mean"]
        covariance = self.posterior["covariance"]
        for start in range(0, len(indices), chunk):
            design = self.design(indices[start:start+chunk].to(device))
            means.append((design@mean_core).cpu())
            variances.append(((design@covariance)*design).sum(1).cpu())
        return torch.cat(means), (torch.cat(variances)+self.posterior["noise"]**2).sqrt()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/the_well_acoustic_64x64"))
    parser.add_argument("--train-cases", type=int, default=8)
    parser.add_argument("--validation-cases", type=int, default=2)
    parser.add_argument("--ratio", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--graph-modes", type=int, default=64)
    parser.add_argument("--time-modes", type=int, default=40)
    parser.add_argument("--output", type=Path,
                        default=Path("papers/longterm_results/the_well_graphmode_a.json"))
    args = parser.parse_args()
    started_basis = time.perf_counter()
    cases=[]; bases=[]; eigenvalues=[]; operator_symmetry=[]; eigen_residuals=[]
    paths = ([args.data/"train"/f"trajectory_{i:03d}.npz" for i in range(args.train_cases)] +
             [args.data/"validation"/f"trajectory_{i:03d}.npz" for i in range(args.validation_cases)])
    for path in paths:
        inputs, target = load_the_well_case(path); cases.append((inputs, target))
        basis, values, operator, residual = graph_eigenbasis(
            inputs["speed_of_sound"], args.graph_modes)
        bases.append(basis); eigenvalues.append(values)
        skew = operator-operator.T
        operator_symmetry.append(float(np.max(np.abs(skew.data))) if skew.nnz else 0.)
        eigen_residuals.append(residual)
    flat_basis, flat_values, _, flat_residual = graph_eigenbasis(
        np.ones_like(cases[0][0]["speed_of_sound"]), args.graph_modes, flat=True)
    basis_seconds = time.perf_counter()-started_basis
    graph_bases = torch.stack(bases); graph_eigenvalues = torch.stack(eigenvalues)
    time_basis, time_eigenvalues = cosine_time_basis(cases[0][1].shape[0], args.time_modes)
    observed_indices=[]; observed_values=[]; held_masks=[]
    for instance, (_, target) in enumerate(cases):
        observed = fixed_random_mask(target.shape, args.ratio, args.seed+instance)
        local = np.argwhere(observed)
        node = local[:, 1]*target.shape[2]+local[:, 2]
        observed_indices.append(np.c_[np.full(len(local), instance), local[:, 0], node])
        observed_values.append(target[observed]); held_masks.append(~observed)
    indices = torch.from_numpy(np.concatenate(observed_indices)).long()
    raw_values = torch.from_numpy(np.concatenate(observed_values)).float()
    center, scale = raw_values.mean(), raw_values.std().clamp_min(1e-6)
    values = (raw_values-center)/scale
    configurations = {
        "graph_tucker": (graph_bases, None),
        "wrong_graph_tucker": (graph_bases.roll(1, 0), None),
        "flat_graph_tucker": (flat_basis[None].expand(len(cases), -1, -1).clone(), None),
        "graph_cp": (graph_bases, 15),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows=[]
    for name, (selected_bases, cp_rank) in configurations.items():
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        model=GraphModeFactorization(selected_bases, time_basis, cp_rank=cp_rank).to(device)
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter(); model.fit(indices, values, time_eigenvalues,
            graph_eigenvalues if name != "flat_graph_tucker" else flat_values[None].expand(len(cases), -1),
            steps=args.steps, seed=args.seed)
        case_metrics=[]
        for instance in range(args.train_cases, len(cases)):
            target=torch.from_numpy(cases[instance][1]); local=torch.cartesian_prod(
                torch.arange(target.shape[0]),torch.arange(target.shape[1]),torch.arange(target.shape[2]))
            node=local[:,1]*target.shape[2]+local[:,2]
            query=torch.stack([torch.full((len(local),),instance),local[:,0],node],1)
            prediction,std=model.predict(query); prediction=prediction.reshape(target.shape)*scale+center
            held=torch.from_numpy(held_masks[instance]); error=prediction[held]-target[held]
            case_metrics.append(float(error.square().mean().sqrt()/target[held].std().clamp_min(1e-8)))
        rows.append({"model":name,"parameters":sum(p.numel() for p in model.parameters()),
                     "validation_macro_nrmse":float(np.mean(case_metrics)),
                     "case_nrmse":case_metrics,"elapsed_seconds":time.perf_counter()-started,
                     "peak_gpu_bytes":int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
                     "posterior_noise":model.posterior["noise"],"posterior_alpha":model.posterior["alpha"]})
        print(f"{name}: validation NRMSE={rows[-1]['validation_macro_nrmse']:.4f}",flush=True)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    payload={"experiment_id":"A-METHOD-R6-WELLMAZE-GRAPHMODE-01-RANDOM","status":"SMOKE",
             "config":vars(args),"basis_seconds":basis_seconds,
             "max_operator_symmetry_error":max(operator_symmetry),
             "max_graph_eigen_residual":max(eigen_residuals),
             "flat_eigen_residual":flat_residual,
             "observed":len(indices),"realized_ratio":float(len(indices)/(len(cases)*np.prod(cases[0][1].shape))),
             "tensor_semantics":["instance/source","time","geometry-specific graph spectral mode"],
             "rows":rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2,default=str))


if __name__=="__main__": main()
