import numpy
from pyscf.pbc import df
from pyscf.pbc import tools

def get_int2e_ip1(mydf):
    cell = mydf.cell
    mesh = mydf.mesh
    nao = cell.nao_nr()

    coord = numpy.asarray(mydf.grids.coords)
    weigh = numpy.asarray(mydf.grids.weights)
    ng = len(weigh)

    from pyscf.pbc.dft.numint import eval_ao
    ao   = eval_ao(cell, coord, deriv=1)
    phi  = numpy.asarray(ao[0])
    dphi = numpy.asarray(ao[1:])
    coul = tools.get_coulG(cell, mesh=mesh)

    assert dm.shape == (nao, nao)
    assert phi.shape == (ng, nao)
    assert dphi.shape == (3, ng, nao)
    assert coul.size == ng

    rho = numpy.einsum("gm,gn->mng", phi.conj(), phi)
    rho_g = tools.fft(rho.reshape(-1, ng), mesh)
    v_g = coul * rho_g
    v_r = tools.ifft(v_g, mesh).real
    v_r *= cell.vol / ng
    v_r = v_r.reshape(nao, nao, ng)
    return numpy.einsum("xgm,gn,klg->xmnkl", dphi, phi.conj(), v_r)

def _get_jk(cell, intor, comp, aosym, script_dms, shls_slice, cintopt=None, vhfopt=None):
    """
    This function is a replacement for the _get_jk function in pyscf.hessian.rhf.
    Use the full (3, nao, nao, nao, nao) eri tensor to compute the given 
    script_dms. It only supports intor='int2e_ip1' with some specific script_dms.
    """

    assert intor == 'int2e_ip1'
    assert comp == 3
    assert aosym == 's1'

    assert cintopt is None
    assert vhfopt is None

    # deal with the script_dms
    assert len(script_dms) % 2 == 0

    einsum_str = []
    for s, dm in script_dms:
        assert "->s1" in s
        s1, s2 = s.split('->s1')
        assert len(s1) == 2
        assert len(s2) == 2
        einsum_str.append("xijkl,%s->x%s" % (s1, s2))

    int2e_ip1 = get_int2e_ip1(cell)

    res = []
    for e in einsum_str:
        res.append(numpy.einsum(e, int2e_ip1))