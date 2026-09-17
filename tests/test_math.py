import numpy as np
from pixorb.geometry import project, residuals
from pixorb.refine import grid_anms

def test_project_identity():
    p=np.array([[1.,2.],[10.,20.]])
    assert np.allclose(project(np.eye(3),p),p)

def test_grid_anms_spreads():
    matches=[(x,y,x,y,1.0) for y in [10,50,90] for x in [10,50,90]]
    out=grid_anms(matches,(100,100),grid=(3,3),per_cell=1)
    assert len(out)==9
