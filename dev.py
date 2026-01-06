import bmap as npb
import numpy as np

from bmap import buffer_symbols


def test1():
    m,n,r,z,t1,t2,ail=buffer_symbols('a','b','u','y','type_ota','type_naa','ail')
    dist1=npb.ar_spec((m,),t1,name='h1') #distinct array 1
    dist2=npb.ar_spec((m,n,5),np.float32,name='h2',) #distinct array 2
    dist3=npb.ar_spec((n,z),int,name='h3',) #distinct array 3
    
    #scratch arrays.
    s1=npb.ar_spec((z,m,3),t1,name='sysr')
    s2=npb.ar_spec((r,z,50),t1,name='uxa',align_ldim=ail)
    s3=npb.ar_spec((m,r),t2,name='ail20',order='F',align_ldim=npb.BufferAlign.AVX)
    
    o1=npb.ar_spec((z,m,'4*y'),t1,name='f95')
    o2=npb.ar_spec((z,m,r),t1,name='fi9',)#align_ldim=npb.BufferAlign.AVX512)
    o3=npb.ar_spec((z,m,'5*u*y'),t1,name='re',order='F')
    
    mem_map=npb.db_node(dist1,dist2,dist3,name='Algo Mem')
    shared_mem=npb.sb_node(s1,s2,s3,name='Shared Mem')
    shared_mem2=npb.sb_node(o1,o2,o3,name='Shared Mem 2')
    mem_map.add(shared_mem,shared_mem2)
    
    print(mem_map,'\n')
    
    rgs=npb.build_bmap(mem_map,align=ail)
    
    print('Built buffer map.',rgs,'\n')
    
    print(mem_map,'\n')
    
    print(npb.build_buffer_allocator(mem_map,rgs,chkforbuffer=True),'\n')

import sympy as sym
def lars1_test():
    #if maxcalc_dims <= 0: maxcalc_dims = min(sample_size
    m,n,dt,il=buffer_symbols('sample_size sample_dims type_flt alignb')
    #ar_spec = npb.arspec_i(dtype=defdtype)
    At = npb.ar_spec((m, m),dt,name='At')
    T1 = npb.ar_spec((sym.Max(m * m, n),),dt,name='T1')
    T2 = npb.ar_spec((m,),dt,name='T2')
    T3 = npb.ar_spec((m,),dt,name='T3')
    C = npb.ar_spec((n,),dt,name='C')
    I = npb.ar_spec((m,), np.int64,name='I')
    Ib = npb.ar_spec((n,), np.bool_,name='Ib')
    spec = npb.db_node(At, T1, T2, T3, C, I, Ib,name='lars1_memspec',no_merge=True)
    
    rgs=npb.build_bmap(spec,align=il)
    # print(rgs)
    # rgs.remove(il)
    #{il:npb.BufferAlign.AVX512}
    print(npb.build_buffer_allocator(spec,rgs,{il:npb.BufferAlign.AVX512},chkforbuffer=True))
    
    
if __name__=="__main__":
    test1()
    lars1_test()
