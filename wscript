import os.path
from waflib import Task
from waflib.Tools import ccroot, c_preproc, cxx
from waflib.Configure import conf

top = '.'
out = 'build'

class cuda(Task.Task):
    run_str = '${NVCC} ${CUDAFLAGS} ${CXXFLAGS} ${CPPPATH_ST:INCPATHS} ${DEFINES_ST:DEFINES} -c ${SRC} -o ${TGT}'
    ext_in = '.cu'
    ext_out = '.o'
    color = 'GREEN'
    vars = ['CCDEPS']
    scan = c_preproc.scan
    shell = False

class cudaprogram(cxx.cxxprogram):
    run_str = '${NVCC} ${CUDAFLAGS} ${LINKFLAGS} ${CXXLNK_SRC_F}${SRC} ${CXXLNK_TGT_F}${TGT[0].abspath()} ${RPATH_ST:RPATH} ${FRAMEWORKPATH_ST:FRAMEWORKPATH} ${FRAMEWORK_ST:FRAMEWORK} ${ARCH_ST:ARCH} ${STLIB_MARKER} ${STLIBPATH_ST:STLIBPATH} ${STLIB_ST:STLIB} ${SHLIB_MARKER} ${LIBPATH_ST:LIBPATH} ${LIB_ST:LIB}'

def options(ctx):
    ctx.load('compiler_cxx')
    ctx.add_option('--with-clogs', action = 'store', help = 'Path to CLOGS')
    ctx.add_option('--with-compute', action = 'store', help = 'Path to Boost.Compute')
    ctx.add_option('--with-vexcl', action = 'store', help = 'Path to VexCL')
    ctx.add_option('--with-thrust', action = 'store', help = 'Path to Thrust')
    ctx.add_option('--with-cub', action = 'store', help = 'Path to CUB')

def check_library(func, option, includes_add, libpath_add, *args, **kw):
    if option is not None:
        if includes_add is not None:
            kw['includes'] = os.path.join(option, includes_add)
        if libpath_add is not None:
            kw['libpath'] = os.path.join(option, libpath_add)
        return func(*args, **kw)
    else:
        return func(*args, mandatory = False, **kw)

def configure(ctx):
    ctx.load('compiler_cxx')

    ctx.env.append_value('CXXFLAGS', ['-Wall', '-O3', '-std=c++11', '-fopenmp'])
    ctx.env.append_value('LINKFLAGS', ['-fopenmp'])

    ctx.find_program('nvcc', var = 'NVCC', mandatory = False)
    ctx.env.have_clogs = check_library(ctx.check_cxx, ctx.options.with_clogs, 'include', 'lib',
        header_name = 'clogs/clogs.h', lib = ['clogs', 'OpenCL'], uselib_store = 'CLOGS')
    ctx.env.have_compute = check_library(ctx.check_cxx, ctx.options.with_compute, 'include', None,
        header_name = 'boost/compute.hpp', lib = ['OpenCL'], uselib_store = 'COMPUTE')
    ctx.env.have_vexcl = check_library(ctx.check_cxx, ctx.options.with_vexcl, '', None,
        header_name = 'vexcl/vexcl.hpp', lib = ['boost_system', 'OpenCL'], uselib_store = 'VEXCL')
    ctx.check_cxx(
        header_name = 'boost/program_options.hpp',
        lib = 'boost_program_options',
        uselib_store = 'PROGRAM_OPTIONS')

def build(ctx):
    sources = ['scanbench.cpp', 'scanbench_cpu.cpp']
    use = ['PROGRAM_OPTIONS']
    if ctx.env.have_clogs:
        sources += ['scanbench_clogs.cpp']
        use += ['CLOGS']
    if ctx.env.have_compute:
        sources += ['scanbench_compute.cpp']
        use += ['COMPUTE']
    if ctx.env.have_vexcl:
        sources += ['scanbench_vex.cpp']
        use += ['VEXCL']
    features = ['cxx']
    features += ['cxxprogram']
    ctx(features = features, source = sources, target = 'scanbench', use = use)
