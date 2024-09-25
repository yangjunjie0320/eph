import numpy
from pyscf.pbc import df
from pyscf.pbc import tools

def get_int2e_ip1(cell):
    mydf = df.FFTDF(cell)
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

    ip_eri = numpy.einsum("xgm,gn,klg->xmnkl", dphi, phi.conj(), v_r, optimize=True)
    return ip_eri

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
    coul = tools.get_coulG(cell, mesh=mesh)

    rho = numpy.einsum("gm,gn->mng", phi.conj(), phi)
    rho_g = tools.fft(rho.reshape(-1, ng), mesh)
    v_g = coul * rho_g
    v_r = tools.ifft(v_g, mesh).real
    v_r *= cell.vol / ng
    v_r = v_r.reshape(nao, nao, ng)

    eri = numpy.einsum("gm,gn,klg->mnkl", phi, phi.conj(), v_r, optimize=True)
    return eri

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

    int2e_ip1 = get_int2e_ip1(cell)
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
        s  = script_dms[i].split('->s1')
        dm = script_dms[i+1]

        einsum_expr = "xijkl,%s->x%s" % (s[0], s[1])  
           
        res.append(
            numpy.einsum(einsum_expr, int2e_ip1[:, p0:p1], dm)
        )

    return res

from pyscf import lib
from pyscf.pbc.df.df_jk import _format_dms
from pyscf.pbc.lib.kpts_helper import is_zero
def _get_j1_ji_kl(df_obj, p0p1=None, dm=None):
    """
    Compute the first order Fock matrix in the AO basis.
    Equivalent to the einsum string:
        xijkl,ji->xkl
    in which xijkl is the full (3, nao, nao, nao, nao) 
    ip_eri tensor.
    """
    cell = df_obj.cell
    mesh = df_obj.mesh
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension > 1

    coul_g = tools.get_coulG(cell, mesh=mesh)

    nao = cell.nao_nr()
    ng = len(coul_g)
    p0, p1 = p0p1
    assert dm.shape == (nao, p1 - p0)

    rho_r = numpy.zeros((3, ng))
    for tmp, g0, g1 in df_obj.aoR_loop(df_obj.grids, deriv=1):
        phi = tmp[0][0]
        assert phi.shape == (4, g1 - g0, nao)

        phi1 = phi[1:, :, p0:p1]
        phi2 = phi[0]

        rho_r[:, g0:g1] += numpy.einsum(
            "xgi,gj,ji->xg",
            phi1, phi2, dm,
            optimize=True
        )

    rho_g = tools.fft(rho_r.reshape(-1, ng), mesh)
    v_g = rho_g * coul_g
    v_r = tools.ifft(v_g, mesh).real
    v_r = v_r.reshape(3, ng)

    weight = cell.vol / ng
    v_r = v_r * weight

    vj1 = numpy.zeros((3, nao, nao))

    for tmp, g0, g1 in df_obj.aoR_loop(df_obj.grids, deriv=0):
        phi = tmp[0][0]
        assert phi.shape == (g1 - g0, nao)

        phi1 = phi.conj()
        phi2 = phi

        vj1 += numpy.einsum(
            "gk,gl,xg->xkl",
            phi1, phi2, v_r,
            optimize=True
        )

    return vj1

def _get_j1_lk_ij(df_obj, p0p1=None, dm=None):
    """
    Compute the first order Fock matrix in the AO basis.
    Equivalent to the einsum string:
        xijkl,lk->xij
    in which xijkl is the full (3, p0:p1, nao, nao, nao) 
    ip_eri tensor.
    """
    cell = df_obj.cell
    mesh = df_obj.mesh
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension > 1

    coul_g = tools.get_coulG(cell, mesh=mesh)

    nao = cell.nao_nr()
    ng = len(coul_g)
    p0, p1 = p0p1
    assert dm.shape == (nao, nao)

    rho_r = numpy.zeros((ng, ))
    for tmp, g0, g1 in df_obj.aoR_loop(df_obj.grids, deriv=0):
        phi = tmp[0][0]
        assert phi.shape == (g1 - g0, nao)

        phi1 = phi2 = phi
        rho_r[g0:g1] += numpy.einsum(
            "gk,gl,lk->g",
            phi1, phi2, dm,
            optimize=True
        )

    rho_g = tools.fft(rho_r.reshape(-1, ng), mesh)
    v_g = rho_g * coul_g
    v_r = tools.ifft(v_g, mesh).real
    v_r = v_r.reshape(ng)

    weight = cell.vol / ng
    v_r = v_r * weight

    vj1 = numpy.zeros((3, p1 - p0, nao))

    for tmp, g0, g1 in df_obj.aoR_loop(df_obj.grids, deriv=1):
        phi = tmp[0][0]
        assert phi.shape == (4, g1 - g0, nao)

        phi1 = phi[1:, :, p0:p1]
        phi2 = phi[0]

        vj1 += numpy.einsum(
            "xgi,gj,g->xij",
            phi1, phi2, v_r,
            optimize=True
        )

    return vj1

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
    cell.precision = 1e-6
    cell.exp_to_discard = 0.1
    cell.build()

    nao = cell.nao_nr()
    df_obj = df.FFTDF(cell)

    int2e_ref = df_obj.get_eri(compact=False).reshape(nao * nao, nao * nao)
    int2e_sol = get_int2e(df_obj).reshape(nao * nao, nao * nao)

    # print("\nint2e_ref")
    # numpy.savetxt(cell.stdout, int2e_ref, fmt='% 6.4f', delimiter=', ')
    # print("\nint2e_sol")
    # numpy.savetxt(cell.stdout, int2e_sol, fmt='% 6.4f', delimiter=', ')

    assert numpy.allclose(int2e_ref, int2e_sol), "int2e_ref and int2e_sol are not close"

    dm0 = numpy.random.rand(1, nao, nao)
    dm0 += dm0.transpose(0, 2, 1)

    j1_ref, k1_ref = df_obj.get_jk_e1(
        dm=dm0, kpts=None, 
        kpts_band=None, exxdiv=None
    )

    j1_ref = j1_ref.reshape(3, nao, nao)
    k1_ref = k1_ref.reshape(3, nao, nao)

    nbas = cell.nbas
    j1_sol = numpy.zeros_like(j1_ref)
    k1_sol = numpy.zeros_like(k1_ref)

    for ia, (s0, s1, p0, p1) in enumerate(cell.aoslice_by_atom()):
        dm0 = dm0.reshape(nao, nao)
        script_dms  = ['lk->s1ij', -dm0]
        script_dms += ['ji->s1kl', -dm0[:, p0:p1]]
        script_dms += ['jk->s1il', -dm0]
        shls_slice = (s0, s1) + (0, nbas) * 3

        tmp = _get_jk(
            cell, 'int2e_ip1', 3, 's1',
            script_dms=script_dms,
            shls_slice=shls_slice
        )

        vj1_ref = tmp[0]
        vj1_sol = _get_j1_lk_ij(df_obj, (p0, p1), -dm0)

        vj2_ref = tmp[1]
        vj2_sol = _get_j1_ji_kl(df_obj, (p0, p1), -dm0[:, p0:p1])

        assert numpy.allclose(vj1_ref, vj1_sol)
        assert numpy.allclose(vj2_ref, vj2_sol)
        
        j1_sol[:, p0:p1, :] += tmp[0]
        k1_sol[:, p0:p1, :] += tmp[2]

    assert numpy.allclose(j1_ref, j1_sol)
    assert numpy.allclose(k1_ref, k1_sol)