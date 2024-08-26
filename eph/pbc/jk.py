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
    assert ao.shape == (4, ng, nao)

    phi  = numpy.asarray(ao[0])
    dphi = numpy.asarray(ao[1:])
    coul = tools.get_coulG(cell, mesh=mesh)

    # assert dm.shape == (nao, nao)
    assert phi.shape == (ng, nao)
    assert dphi.shape == (3, ng, nao)
    assert coul.size == ng

    rho = numpy.einsum("gm,gn->mng", phi.conj(), phi)
    rho_g = tools.fft(rho.reshape(-1, ng), mesh)
    v_g = coul * rho_g
    v_r = tools.ifft(v_g, mesh).real
    v_r *= cell.vol / ng
    v_r = v_r.reshape(nao, nao, ng)
    return numpy.einsum("xgm,gn,klg->xmnkl", dphi, phi.conj(), v_r, optimize=True)

def get_int2e(mydf):
    cell = mydf.cell
    mesh = mydf.mesh
    nao = cell.nao_nr()

    coord = numpy.asarray(mydf.grids.coords)
    weigh = numpy.asarray(mydf.grids.weights)
    ng = len(weigh)

    from pyscf.pbc.dft.numint import eval_ao
    ao   = eval_ao(cell, coord, deriv=1)
    assert ao.shape == (4, ng, nao)
    
    phi  = numpy.asarray(ao[0])
    dphi = numpy.asarray(ao[1:])
    coul = tools.get_coulG(cell, mesh=mesh)

    rho = numpy.einsum("gm,gn->mng", phi.conj(), phi)
    rho_g = tools.fft(rho.reshape(-1, ng), mesh)
    v_g = coul * rho_g
    v_r = tools.ifft(v_g, mesh).real
    v_r *= cell.vol / ng
    v_r = v_r.reshape(nao, nao, ng)
    return numpy.einsum("gm,gn,klg->mnkl", phi, phi.conj(), v_r, optimize=True)

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

    df_obj = df.FFTDF(cell)
    int2e_ip1 = get_int2e_ip1(df_obj)
    nao = cell.nao_nr()

    # deal with the script_dms
    assert len(script_dms) % 2 == 0
    s0, s1, p0, p1 = (None,) * 4
    for (s0, s1, p0, p1) in cell.aoslice_by_atom():
        if s0 == shls_slice[0] and s1 == shls_slice[1]:
            break

    assert s0 is not None
    assert s1 is not None
    assert p0 is not None
    assert p1 is not None

    res = []
    for i in range(0, len(script_dms), 2):
        s  = script_dms[i]
        dm = script_dms[i+1]

        assert "->s1" in s, "Only '->s1' is supported: %s" % s
        s1, s2 = s.split('->s1')

        assert len(s1) == 2
        assert len(s2) == 2

        res.append(
            numpy.einsum(
                "xijkl,%s->x%s" % (s1, s2),
                int2e_ip1[:, p0:p1], dm, optimize=True
            )
        )

    return res

if __name__ == "__main__":
    from pyscf.pbc import gto
    cell = gto.Cell()
    cell.atom = '''
    Li 1.000000 1.000000 1.000000
    Li 1.000000 1.000000 2.000000
    '''
    cell.a = numpy.diag([2.0, 2.0, 3.0])
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.unit = 'A'
    cell.verbose = 0
    cell.ke_cutoff = 100
    cell.exp_to_discard = 0.1
    cell.build()

    nao = cell.nao_nr()
    df_obj = df.FFTDF(cell)

    int2e_ref = df_obj.get_eri(compact=False).reshape(nao, nao, nao, nao)
    int2e_sol = get_int2e(df_obj).reshape(nao, nao, nao, nao)

    assert numpy.allclose(int2e_ref, int2e_sol)

    dm0 = numpy.random.rand(1, nao, nao)
    dm0 += dm0.transpose(0, 2, 1)

    j1_ref, k1_ref = df_obj.get_jk_e1(
        dm=dm0, kpts=None, kpts_band=None,
        exxdiv=None
    )

    j1_ref = j1_ref.reshape(3, nao, nao)
    k1_ref = k1_ref.reshape(3, nao, nao)

    nbas = cell.nbas
    j1_sol = numpy.zeros_like(j1_ref)
    k1_sol = numpy.zeros_like(k1_ref)

    for ia, (s0, s1, p0, p1) in enumerate(cell.aoslice_by_atom()):
        dm0 = dm0.reshape(nao, nao)
        script_dms  = ['lk->s1ij', -dm0]
        script_dms += ['jk->s1il', -dm0]
        shls_slice = (s0, s1) + (0, nbas) * 3

        tmp = _get_jk(
            cell, 'int2e_ip1', 3, 's1',
            script_dms=script_dms,
            shls_slice=shls_slice
        )
        
        j1_sol[:, p0:p1, :] += tmp[0]
        k1_sol[:, p0:p1, :] += tmp[1]

    assert numpy.allclose(j1_ref, j1_sol)
    assert numpy.allclose(k1_ref, k1_sol)