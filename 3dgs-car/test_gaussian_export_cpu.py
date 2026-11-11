"""Exercise the real export function and main export gate without CUDA imports."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch


@pytest.mark.parametrize('cache_only, requested, expect_npz, expect_checkpoint', [
    (True, True, True, False), (True, False, False, False),
    (False, False, True, True),
])
def test_gaussian_export_in_cache_mode(tmp_path, cache_only, requested, expect_npz, expect_checkpoint):
    source=ast.parse(Path(__file__).with_name('train_stage2_npz.py').read_text())
    export=next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name=='export_gaussians')
    main=next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name=='main')
    gate=next(n for n in main.body if isinstance(n, ast.If) and 'args.save_gaussian_npz' in ast.unparse(n.test))
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),export,gate],type_ignores=[])
    ast.fix_missing_locations(module)
    case=SimpleNamespace(sample_name='rca_0452',path=Path('rca_0452.npz'),
        theta_deg=np.array([10.,20.]),phi_deg=np.array([80.,80.]),projection_center_offset_m=np.zeros(3))
    gaussians=SimpleNamespace(get_xyz=torch.tensor([[.5,.5,.5]]),get_density=torch.ones((1,1)),
        get_scaling=torch.ones((1,3))*.01,get_rotation=torch.tensor([[1.,0.,0.,0.]]))
    namespace=dict(np=np,torch=torch,cpu_state_dict=lambda model:{},
        args=SimpleNamespace(evaluation_cache_only=cache_only,save_gaussian_npz=requested),
        gaussians=gaussians,output_dir=tmp_path,case=case,view_indices=np.array([0,1]),
        cone_vectors=np.zeros((2,12)),evaluated_theta_deg=case.theta_deg,
        evaluated_phi_deg=case.phi_deg+.2,volume_extent_m=.12,silhouette_gain=1.)
    exec(compile(module,'train_stage2_npz_export','exec'),namespace)
    assert (tmp_path/'gaussians.npz').exists()==expect_npz
    assert (tmp_path/'gaussians.pt').exists()==expect_checkpoint
    if expect_npz:
        with np.load(tmp_path/'gaussians.npz',allow_pickle=False) as data:
            np.testing.assert_array_equal(data['view_indices'],[0,1])
            np.testing.assert_allclose(data['evaluated_phi_deg'],[80.2,80.2])
            assert data['xyz_world_m'].shape==(1,3)
