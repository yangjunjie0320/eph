import pyscf
from pyscf.pbc.scf.hf import SCF

def _is_pbc(mf_obj):
    is_pbc = isinstance(mf_obj, pyscf.pbc.scf.hf.SCF)
    is_cell = isinstance(mf_obj.mol, pyscf.pbc.gto.Cell)
    assert is_pbc == is_cell
    return is_pbc

def get_jk1(mf_obj, script_dms, shls_slice):
    if not _is_pbc(mf_obj):
        from pyscf.hessian.rhf import _get_jk
        mol   = mf_obj.mol
        intor = 'int2e_ip1'

        assert len(script_dms) % 2 == 0
        res = _get_jk(
            mol, intor, 3, 's1',
            script_dms=script_dms,
            shls_slice=shls_slice
        )

    else:
        raise NotImplementedError
    
    return res

def gen_hcore_deriv(mf_obj):
    if not _is_pbc(mf_obj):
        return mf_obj.nuc_grad_method().hcore_generator()
    else:
        raise NotImplementedError
    
def get_ipovlp(mf_obj):
    if not _is_pbc(mf_obj):
        mol = mf_obj.mol
        return mol.intor("int1e_ipovlp")
    else:
        cell = mf_obj.cell
        return cell.pbc_intor("int1e_ipovlp")