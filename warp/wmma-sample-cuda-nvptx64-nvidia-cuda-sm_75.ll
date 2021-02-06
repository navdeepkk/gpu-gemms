; ModuleID = 'wmma-sample.cu'
source_filename = "wmma-sample.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.__half = type { i16 }
%struct.__cuda_builtin_threadIdx_t = type { i8 }
%struct.cudaFuncAttributes = type { i64, i64, i64, i32, i32, i32, i32, i32, i32, i32 }
%"class.nvcuda::wmma::fragment" = type { %"struct.nvcuda::wmma::__frag_base" }
%"struct.nvcuda::wmma::__frag_base" = type { [16 x %struct.__half] }
%"class.nvcuda::wmma::fragment.0" = type { %"struct.nvcuda::wmma::__frag_base.1" }
%"struct.nvcuda::wmma::__frag_base.1" = type { [8 x %struct.__half] }
%"class.nvcuda::wmma::fragment.2" = type { %"struct.nvcuda::wmma::__frag_base" }

$_ZN6nvcuda4wmma8fragmentINS0_8matrix_aELi16ELi16ELi16E6__halfNS0_9row_majorEEC1Ev = comdat any

$_ZN6nvcuda4wmma8fragmentINS0_11accumulatorELi16ELi16ELi16E6__halfvEC1Ev = comdat any

$_ZN6__halfC1Ef = comdat any

$_ZN6nvcuda4wmma8fragmentINS0_8matrix_bELi16ELi16ELi16E6__halfNS0_9col_majorEEC1Ev = comdat any

$_ZN6nvcuda4wmma8fragmentINS0_8matrix_aELi16ELi16ELi16E6__halfNS0_9row_majorEEC2Ev = comdat any

$_ZN6nvcuda4wmma11__frag_baseI6__halfLi16ELi16EEC2Ev = comdat any

$_ZN6__halfC1Ev = comdat any

$_ZN6__halfC2Ev = comdat any

$_ZN6nvcuda4wmma8fragmentINS0_11accumulatorELi16ELi16ELi16E6__halfvEC2Ev = comdat any

$_ZN6nvcuda4wmma11__frag_baseI6__halfLi8ELi8EEC2Ev = comdat any

$_ZN6__halfC2Ef = comdat any

$_ZN6nvcuda4wmma8fragmentINS0_8matrix_bELi16ELi16ELi16E6__halfNS0_9col_majorEEC2Ev = comdat any

@_ZZ9test_wmmavE4smem = internal addrspace(3) global [8192 x %struct.__half] undef, align 2
@threadIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_threadIdx_t, align 1

; Function Attrs: convergent noinline nounwind optnone
define weak dso_local i32 @cudaMalloc(i8** %p, i64 %s) #0 {
entry:
  %p.addr = alloca i8**, align 8
  %s.addr = alloca i64, align 8
  store i8** %p, i8*** %p.addr, align 8
  store i64 %s, i64* %s.addr, align 8
  ret i32 999
}

; Function Attrs: convergent noinline nounwind optnone
define weak dso_local i32 @cudaFuncGetAttributes(%struct.cudaFuncAttributes* %p, i8* %c) #0 {
entry:
  %p.addr = alloca %struct.cudaFuncAttributes*, align 8
  %c.addr = alloca i8*, align 8
  store %struct.cudaFuncAttributes* %p, %struct.cudaFuncAttributes** %p.addr, align 8
  store i8* %c, i8** %c.addr, align 8
  ret i32 999
}

; Function Attrs: convergent noinline nounwind optnone
define weak dso_local i32 @cudaDeviceGetAttribute(i32* %value, i32 %attr, i32 %device) #0 {
entry:
  %value.addr = alloca i32*, align 8
  %attr.addr = alloca i32, align 4
  %device.addr = alloca i32, align 4
  store i32* %value, i32** %value.addr, align 8
  store i32 %attr, i32* %attr.addr, align 4
  store i32 %device, i32* %device.addr, align 4
  ret i32 999
}

; Function Attrs: convergent noinline nounwind optnone
define weak dso_local i32 @cudaGetDevice(i32* %device) #0 {
entry:
  %device.addr = alloca i32*, align 8
  store i32* %device, i32** %device.addr, align 8
  ret i32 999
}

; Function Attrs: convergent noinline nounwind optnone
define weak dso_local i32 @cudaOccupancyMaxActiveBlocksPerMultiprocessor(i32* %numBlocks, i8* %func, i32 %blockSize, i64 %dynamicSmemSize) #0 {
entry:
  %numBlocks.addr = alloca i32*, align 8
  %func.addr = alloca i8*, align 8
  %blockSize.addr = alloca i32, align 4
  %dynamicSmemSize.addr = alloca i64, align 8
  store i32* %numBlocks, i32** %numBlocks.addr, align 8
  store i8* %func, i8** %func.addr, align 8
  store i32 %blockSize, i32* %blockSize.addr, align 4
  store i64 %dynamicSmemSize, i64* %dynamicSmemSize.addr, align 8
  ret i32 999
}

; Function Attrs: convergent noinline nounwind optnone
define weak dso_local i32 @cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(i32* %numBlocks, i8* %func, i32 %blockSize, i64 %dynamicSmemSize, i32 %flags) #0 {
entry:
  %numBlocks.addr = alloca i32*, align 8
  %func.addr = alloca i8*, align 8
  %blockSize.addr = alloca i32, align 4
  %dynamicSmemSize.addr = alloca i64, align 8
  %flags.addr = alloca i32, align 4
  store i32* %numBlocks, i32** %numBlocks.addr, align 8
  store i8* %func, i8** %func.addr, align 8
  store i32 %blockSize, i32* %blockSize.addr, align 4
  store i64 %dynamicSmemSize, i64* %dynamicSmemSize.addr, align 8
  store i32 %flags, i32* %flags.addr, align 4
  ret i32 999
}

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local void @_Z9test_wmmav() #1 {
entry:
  %A = alloca %struct.__half*, align 8
  %B = alloca %struct.__half*, align 8
  %C = alloca %struct.__half*, align 8
  %a_frag_2 = alloca [10 x %"class.nvcuda::wmma::fragment"], align 4
  %a_frag = alloca %"class.nvcuda::wmma::fragment", align 4
  %acc_frag = alloca %"class.nvcuda::wmma::fragment.0", align 4
  %ref.tmp = alloca %struct.__half, align 2
  %i = alloca i32, align 4
  %b_frag = alloca %"class.nvcuda::wmma::fragment.2", align 4
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #6, !range !10
  %mul = mul i32 %0, 1024
  %idx.ext = zext i32 %mul to i64
  %add.ptr = getelementptr inbounds %struct.__half, %struct.__half* getelementptr inbounds ([8192 x %struct.__half], [8192 x %struct.__half]* addrspacecast ([8192 x %struct.__half] addrspace(3)* @_ZZ9test_wmmavE4smem to [8192 x %struct.__half]*), i64 0, i64 0), i64 %idx.ext
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #6, !range !10
  %mul2 = mul i32 %1, 16
  %idx.ext3 = zext i32 %mul2 to i64
  %add.ptr4 = getelementptr inbounds %struct.__half, %struct.__half* %add.ptr, i64 %idx.ext3
  store %struct.__half* %add.ptr4, %struct.__half** %A, align 8
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #6, !range !10
  %mul6 = mul i32 %2, 1024
  %idx.ext7 = zext i32 %mul6 to i64
  %add.ptr8 = getelementptr inbounds %struct.__half, %struct.__half* getelementptr inbounds ([8192 x %struct.__half], [8192 x %struct.__half]* addrspacecast ([8192 x %struct.__half] addrspace(3)* @_ZZ9test_wmmavE4smem to [8192 x %struct.__half]*), i64 0, i64 0), i64 %idx.ext7
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #6, !range !10
  %mul10 = mul i32 %3, 16
  %idx.ext11 = zext i32 %mul10 to i64
  %add.ptr12 = getelementptr inbounds %struct.__half, %struct.__half* %add.ptr8, i64 %idx.ext11
  %add.ptr13 = getelementptr inbounds %struct.__half, %struct.__half* %add.ptr12, i64 256
  store %struct.__half* %add.ptr13, %struct.__half** %B, align 8
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #6, !range !10
  %mul15 = mul i32 %4, 1024
  %idx.ext16 = zext i32 %mul15 to i64
  %add.ptr17 = getelementptr inbounds %struct.__half, %struct.__half* getelementptr inbounds ([8192 x %struct.__half], [8192 x %struct.__half]* addrspacecast ([8192 x %struct.__half] addrspace(3)* @_ZZ9test_wmmavE4smem to [8192 x %struct.__half]*), i64 0, i64 0), i64 %idx.ext16
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #6, !range !10
  %mul19 = mul i32 %5, 16
  %idx.ext20 = zext i32 %mul19 to i64
  %add.ptr21 = getelementptr inbounds %struct.__half, %struct.__half* %add.ptr17, i64 %idx.ext20
  %add.ptr22 = getelementptr inbounds %struct.__half, %struct.__half* %add.ptr21, i64 512
  store %struct.__half* %add.ptr22, %struct.__half** %C, align 8
  %array.begin = getelementptr inbounds [10 x %"class.nvcuda::wmma::fragment"], [10 x %"class.nvcuda::wmma::fragment"]* %a_frag_2, i32 0, i32 0
  %arrayctor.end = getelementptr inbounds %"class.nvcuda::wmma::fragment", %"class.nvcuda::wmma::fragment"* %array.begin, i64 10
  br label %arrayctor.loop

arrayctor.loop:                                   ; preds = %arrayctor.loop, %entry
  %arrayctor.cur = phi %"class.nvcuda::wmma::fragment"* [ %array.begin, %entry ], [ %arrayctor.next, %arrayctor.loop ]
  call void @_ZN6nvcuda4wmma8fragmentINS0_8matrix_aELi16ELi16ELi16E6__halfNS0_9row_majorEEC1Ev(%"class.nvcuda::wmma::fragment"* nonnull dereferenceable(32) %arrayctor.cur) #7
  %arrayctor.next = getelementptr inbounds %"class.nvcuda::wmma::fragment", %"class.nvcuda::wmma::fragment"* %arrayctor.cur, i64 1
  %arrayctor.done = icmp eq %"class.nvcuda::wmma::fragment"* %arrayctor.next, %arrayctor.end
  br i1 %arrayctor.done, label %arrayctor.cont, label %arrayctor.loop

arrayctor.cont:                                   ; preds = %arrayctor.loop
  call void @_ZN6nvcuda4wmma8fragmentINS0_8matrix_aELi16ELi16ELi16E6__halfNS0_9row_majorEEC1Ev(%"class.nvcuda::wmma::fragment"* nonnull dereferenceable(32) %a_frag) #7
  call void @_ZN6nvcuda4wmma8fragmentINS0_11accumulatorELi16ELi16ELi16E6__halfvEC1Ev(%"class.nvcuda::wmma::fragment.0"* nonnull dereferenceable(16) %acc_frag) #7
  %6 = bitcast %"class.nvcuda::wmma::fragment.0"* %acc_frag to %"struct.nvcuda::wmma::__frag_base.1"*
  call void @_ZN6__halfC1Ef(%struct.__half* nonnull dereferenceable(2) %ref.tmp, float 0.000000e+00) #7
  call void @_ZN6nvcuda4wmmaL13fill_fragmentI6__halfLi8ELi8EEEvRNS0_11__frag_baseIT_XT0_EXT1_EEERKNS0_13helper_traitsIS4_E18fill_argument_typeE(%"struct.nvcuda::wmma::__frag_base.1"* nonnull align 4 dereferenceable(16) %6, %struct.__half* nonnull align 2 dereferenceable(2) %ref.tmp) #7
  %7 = load %struct.__half*, %struct.__half** %A, align 8
  call void @_ZN6nvcuda4wmmaL16load_matrix_syncERNS0_8fragmentINS0_8matrix_aELi16ELi16ELi16E6__halfNS0_9row_majorEEEPKS3_j(%"class.nvcuda::wmma::fragment"* nonnull align 4 dereferenceable(32) %a_frag, %struct.__half* %7, i32 16) #7
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %arrayctor.cont
  %8 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %8, 20
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  call void @_ZN6nvcuda4wmma8fragmentINS0_8matrix_bELi16ELi16ELi16E6__halfNS0_9col_majorEEC1Ev(%"class.nvcuda::wmma::fragment.2"* nonnull dereferenceable(32) %b_frag) #7
  %9 = load %struct.__half*, %struct.__half** %B, align 8
  call void @_ZN6nvcuda4wmmaL16load_matrix_syncERNS0_8fragmentINS0_8matrix_bELi16ELi16ELi16E6__halfNS0_9col_majorEEEPKS3_j(%"class.nvcuda::wmma::fragment.2"* nonnull align 4 dereferenceable(32) %b_frag, %struct.__half* %9, i32 16) #7
  call void @_ZN6nvcuda4wmmaL8mma_syncERNS0_8fragmentINS0_11accumulatorELi16ELi16ELi16E6__halfvEERKNS1_INS0_8matrix_aELi16ELi16ELi16ES3_NS0_9row_majorEEERKNS1_INS0_8matrix_bELi16ELi16ELi16ES3_NS0_9col_majorEEERKS4_(%"class.nvcuda::wmma::fragment.0"* nonnull align 4 dereferenceable(16) %acc_frag, %"class.nvcuda::wmma::fragment"* nonnull align 4 dereferenceable(32) %a_frag, %"class.nvcuda::wmma::fragment.2"* nonnull align 4 dereferenceable(32) %b_frag, %"class.nvcuda::wmma::fragment.0"* nonnull align 4 dereferenceable(16) %acc_frag) #7
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %10 = load i32, i32* %i, align 4
  %inc = add nsw i32 %10, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond, !llvm.loop !11

for.end:                                          ; preds = %for.cond
  %11 = load %struct.__half*, %struct.__half** %C, align 8
  call void @_ZN6nvcuda4wmmaL17store_matrix_syncEP6__halfRKNS0_8fragmentINS0_11accumulatorELi16ELi16ELi16ES1_vEEjNS0_8layout_tE(%struct.__half* %11, %"class.nvcuda::wmma::fragment.0"* nonnull align 4 dereferenceable(16) %acc_frag, i32 16, i32 1) #7
  ret void
}

; Function Attrs: convergent noinline nounwind optnone
define linkonce_odr dso_local void @_ZN6nvcuda4wmma8fragmentINS0_8matrix_aELi16ELi16ELi16E6__halfNS0_9row_majorEEC1Ev(%"class.nvcuda::wmma::fragment"* nonnull dereferenceable(32) %this) unnamed_addr #0 comdat align 2 {
entry:
  %this.addr = alloca %"class.nvcuda::wmma::fragment"*, align 8
  store %"class.nvcuda::wmma::fragment"* %this, %"class.nvcuda::wmma::fragment"** %this.addr, align 8
  %this1 = load %"class.nvcuda::wmma::fragment"*, %"class.nvcuda::wmma::fragment"** %this.addr, align 8
  call void @_ZN6nvcuda4wmma8fragmentINS0_8matrix_aELi16ELi16ELi16E6__halfNS0_9row_majorEEC2Ev(%"class.nvcuda::wmma::fragment"* nonnull dereferenceable(32) %this1) #7
  ret void
}

; Function Attrs: convergent noinline nounwind optnone
define linkonce_odr dso_local void @_ZN6nvcuda4wmma8fragmentINS0_11accumulatorELi16ELi16ELi16E6__halfvEC1Ev(%"class.nvcuda::wmma::fragment.0"* nonnull dereferenceable(16) %this) unnamed_addr #0 comdat align 2 {
entry:
  %this.addr = alloca %"class.nvcuda::wmma::fragment.0"*, align 8
  store %"class.nvcuda::wmma::fragment.0"* %this, %"class.nvcuda::wmma::fragment.0"** %this.addr, align 8
  %this1 = load %"class.nvcuda::wmma::fragment.0"*, %"class.nvcuda::wmma::fragment.0"** %this.addr, align 8
  call void @_ZN6nvcuda4wmma8fragmentINS0_11accumulatorELi16ELi16ELi16E6__halfvEC2Ev(%"class.nvcuda::wmma::fragment.0"* nonnull dereferenceable(16) %this1) #7
  ret void
}

; Function Attrs: convergent noinline nounwind optnone
define internal void @_ZN6nvcuda4wmmaL13fill_fragmentI6__halfLi8ELi8EEEvRNS0_11__frag_baseIT_XT0_EXT1_EEERKNS0_13helper_traitsIS4_E18fill_argument_typeE(%"struct.nvcuda::wmma::__frag_base.1"* nonnull align 4 dereferenceable(16) %f, %struct.__half* nonnull align 2 dereferenceable(2) %in) #0 {
entry:
  %f.addr = alloca %"struct.nvcuda::wmma::__frag_base.1"*, align 8
  %in.addr = alloca %struct.__half*, align 8
  %v = alloca %struct.__half, align 2
  %agg.tmp = alloca %struct.__half, align 2
  %i = alloca i32, align 4
  store %"struct.nvcuda::wmma::__frag_base.1"* %f, %"struct.nvcuda::wmma::__frag_base.1"** %f.addr, align 8
  store %struct.__half* %in, %struct.__half** %in.addr, align 8
  %0 = load %struct.__half*, %struct.__half** %in.addr, align 8
  %1 = bitcast %struct.__half* %agg.tmp to i8*
  %2 = bitcast %struct.__half* %0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 2 %1, i8* align 2 %2, i64 2, i1 false)
  %call = call %struct.__half @_ZN6nvcuda4wmmaL19__get_storage_valueI6__halfS2_S2_EET0_T1_(%struct.__half* byval(%struct.__half) align 2 %agg.tmp) #7
  %3 = getelementptr inbounds %struct.__half, %struct.__half* %v, i32 0, i32 0
  %4 = extractvalue %struct.__half %call, 0
  store i16 %4, i16* %3, align 2
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %5 = load i32, i32* %i, align 4
  %6 = load %"struct.nvcuda::wmma::__frag_base.1"*, %"struct.nvcuda::wmma::__frag_base.1"** %f.addr, align 8
  %cmp = icmp slt i32 %5, 8
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %7 = load %"struct.nvcuda::wmma::__frag_base.1"*, %"struct.nvcuda::wmma::__frag_base.1"** %f.addr, align 8
  %x = getelementptr inbounds %"struct.nvcuda::wmma::__frag_base.1", %"struct.nvcuda::wmma::__frag_base.1"* %7, i32 0, i32 0
  %8 = load i32, i32* %i, align 4
  %idxprom = sext i32 %8 to i64
  %arrayidx = getelementptr inbounds [8 x %struct.__half], [8 x %struct.__half]* %x, i64 0, i64 %idxprom
  %9 = bitcast %struct.__half* %arrayidx to i8*
  %10 = bitcast %struct.__half* %v to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 2 %9, i8* align 2 %10, i64 2, i1 false)
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %11 = load i32, i32* %i, align 4
  %inc = add nsw i32 %11, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond, !llvm.loop !13

for.end:                                          ; preds = %for.cond
  ret void
}

; Function Attrs: convergent noinline nounwind optnone
define linkonce_odr dso_local void @_ZN6__halfC1Ef(%struct.__half* nonnull dereferenceable(2) %this, float %f) unnamed_addr #0 comdat align 2 {
entry:
  %this.addr = alloca %struct.__half*, align 8
  %f.addr = alloca float, align 4
  store %struct.__half* %this, %struct.__half** %this.addr, align 8
  store float %f, float* %f.addr, align 4
  %this1 = load %struct.__half*, %struct.__half** %this.addr, align 8
  %0 = load float, float* %f.addr, align 4
  call void @_ZN6__halfC2Ef(%struct.__half* nonnull dereferenceable(2) %this1, float %0) #7
  ret void
}

; Function Attrs: convergent noinline nounwind optnone
define internal void @_ZN6nvcuda4wmmaL16load_matrix_syncERNS0_8fragmentINS0_8matrix_aELi16ELi16ELi16E6__halfNS0_9row_majorEEEPKS3_j(%"class.nvcuda::wmma::fragment"* nonnull align 4 dereferenceable(32) %a, %struct.__half* %p, i32 %ldm) #0 {
entry:
  %a.addr = alloca %"class.nvcuda::wmma::fragment"*, align 8
  %p.addr = alloca %struct.__half*, align 8
  %ldm.addr = alloca i32, align 4
  store %"class.nvcuda::wmma::fragment"* %a, %"class.nvcuda::wmma::fragment"** %a.addr, align 8
  store %struct.__half* %p, %struct.__half** %p.addr, align 8
  store i32 %ldm, i32* %ldm.addr, align 4
  %0 = load %"class.nvcuda::wmma::fragment"*, %"class.nvcuda::wmma::fragment"** %a.addr, align 8
  %1 = bitcast %"class.nvcuda::wmma::fragment"* %0 to i32*
  %2 = load %struct.__half*, %struct.__half** %p.addr, align 8
  %3 = bitcast %struct.__half* %2 to i32*
  %4 = load i32, i32* %ldm.addr, align 4
  %5 = call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p0i32(i32* %3, i32 %4)
  %6 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %5, 0
  %7 = bitcast <2 x half> %6 to i32
  %8 = getelementptr i32, i32* %1, i32 0
  store i32 %7, i32* %8, align 4
  %9 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %5, 1
  %10 = bitcast <2 x half> %9 to i32
  %11 = getelementptr i32, i32* %1, i32 1
  store i32 %10, i32* %11, align 4
  %12 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %5, 2
  %13 = bitcast <2 x half> %12 to i32
  %14 = getelementptr i32, i32* %1, i32 2
  store i32 %13, i32* %14, align 4
  %15 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %5, 3
  %16 = bitcast <2 x half> %15 to i32
  %17 = getelementptr i32, i32* %1, i32 3
  store i32 %16, i32* %17, align 4
  %18 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %5, 4
  %19 = bitcast <2 x half> %18 to i32
  %20 = getelementptr i32, i32* %1, i32 4
  store i32 %19, i32* %20, align 4
  %21 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %5, 5
  %22 = bitcast <2 x half> %21 to i32
  %23 = getelementptr i32, i32* %1, i32 5
  store i32 %22, i32* %23, align 4
  %24 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %5, 6
  %25 = bitcast <2 x half> %24 to i32
  %26 = getelementptr i32, i32* %1, i32 6
  store i32 %25, i32* %26, align 4
  %27 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %5, 7
  %28 = bitcast <2 x half> %27 to i32
  %29 = getelementptr i32, i32* %1, i32 7
  store i32 %28, i32* %29, align 4
  ret void
}

; Function Attrs: convergent noinline nounwind optnone
define linkonce_odr dso_local void @_ZN6nvcuda4wmma8fragmentINS0_8matrix_bELi16ELi16ELi16E6__halfNS0_9col_majorEEC1Ev(%"class.nvcuda::wmma::fragment.2"* nonnull dereferenceable(32) %this) unnamed_addr #0 comdat align 2 {
entry:
  %this.addr = alloca %"class.nvcuda::wmma::fragment.2"*, align 8
  store %"class.nvcuda::wmma::fragment.2"* %this, %"class.nvcuda::wmma::fragment.2"** %this.addr, align 8
  %this1 = load %"class.nvcuda::wmma::fragment.2"*, %"class.nvcuda::wmma::fragment.2"** %this.addr, align 8
  call void @_ZN6nvcuda4wmma8fragmentINS0_8matrix_bELi16ELi16ELi16E6__halfNS0_9col_majorEEC2Ev(%"class.nvcuda::wmma::fragment.2"* nonnull dereferenceable(32) %this1) #7
  ret void
}

; Function Attrs: convergent noinline nounwind optnone
define internal void @_ZN6nvcuda4wmmaL16load_matrix_syncERNS0_8fragmentINS0_8matrix_bELi16ELi16ELi16E6__halfNS0_9col_majorEEEPKS3_j(%"class.nvcuda::wmma::fragment.2"* nonnull align 4 dereferenceable(32) %a, %struct.__half* %p, i32 %ldm) #0 {
entry:
  %a.addr = alloca %"class.nvcuda::wmma::fragment.2"*, align 8
  %p.addr = alloca %struct.__half*, align 8
  %ldm.addr = alloca i32, align 4
  store %"class.nvcuda::wmma::fragment.2"* %a, %"class.nvcuda::wmma::fragment.2"** %a.addr, align 8
  store %struct.__half* %p, %struct.__half** %p.addr, align 8
  store i32 %ldm, i32* %ldm.addr, align 4
  %0 = load %"class.nvcuda::wmma::fragment.2"*, %"class.nvcuda::wmma::fragment.2"** %a.addr, align 8
  %1 = bitcast %"class.nvcuda::wmma::fragment.2"* %0 to i32*
  %2 = load %struct.__half*, %struct.__half** %p.addr, align 8
  %3 = bitcast %struct.__half* %2 to i32*
  %4 = load i32, i32* %ldm.addr, align 4
  %5 = call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.b.col.stride.f16.p0i32(i32* %3, i32 %4)
  %6 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %5, 0
  %7 = bitcast <2 x half> %6 to i32
  %8 = getelementptr i32, i32* %1, i32 0
  store i32 %7, i32* %8, align 4
  %9 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %5, 1
  %10 = bitcast <2 x half> %9 to i32
  %11 = getelementptr i32, i32* %1, i32 1
  store i32 %10, i32* %11, align 4
  %12 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %5, 2
  %13 = bitcast <2 x half> %12 to i32
  %14 = getelementptr i32, i32* %1, i32 2
  store i32 %13, i32* %14, align 4
  %15 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %5, 3
  %16 = bitcast <2 x half> %15 to i32
  %17 = getelementptr i32, i32* %1, i32 3
  store i32 %16, i32* %17, align 4
  %18 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %5, 4
  %19 = bitcast <2 x half> %18 to i32
  %20 = getelementptr i32, i32* %1, i32 4
  store i32 %19, i32* %20, align 4
  %21 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %5, 5
  %22 = bitcast <2 x half> %21 to i32
  %23 = getelementptr i32, i32* %1, i32 5
  store i32 %22, i32* %23, align 4
  %24 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %5, 6
  %25 = bitcast <2 x half> %24 to i32
  %26 = getelementptr i32, i32* %1, i32 6
  store i32 %25, i32* %26, align 4
  %27 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %5, 7
  %28 = bitcast <2 x half> %27 to i32
  %29 = getelementptr i32, i32* %1, i32 7
  store i32 %28, i32* %29, align 4
  ret void
}

; Function Attrs: convergent noinline nounwind optnone
define internal void @_ZN6nvcuda4wmmaL8mma_syncERNS0_8fragmentINS0_11accumulatorELi16ELi16ELi16E6__halfvEERKNS1_INS0_8matrix_aELi16ELi16ELi16ES3_NS0_9row_majorEEERKNS1_INS0_8matrix_bELi16ELi16ELi16ES3_NS0_9col_majorEEERKS4_(%"class.nvcuda::wmma::fragment.0"* nonnull align 4 dereferenceable(16) %d, %"class.nvcuda::wmma::fragment"* nonnull align 4 dereferenceable(32) %a, %"class.nvcuda::wmma::fragment.2"* nonnull align 4 dereferenceable(32) %b, %"class.nvcuda::wmma::fragment.0"* nonnull align 4 dereferenceable(16) %c) #0 {
entry:
  %d.addr = alloca %"class.nvcuda::wmma::fragment.0"*, align 8
  %a.addr = alloca %"class.nvcuda::wmma::fragment"*, align 8
  %b.addr = alloca %"class.nvcuda::wmma::fragment.2"*, align 8
  %c.addr = alloca %"class.nvcuda::wmma::fragment.0"*, align 8
  store %"class.nvcuda::wmma::fragment.0"* %d, %"class.nvcuda::wmma::fragment.0"** %d.addr, align 8
  store %"class.nvcuda::wmma::fragment"* %a, %"class.nvcuda::wmma::fragment"** %a.addr, align 8
  store %"class.nvcuda::wmma::fragment.2"* %b, %"class.nvcuda::wmma::fragment.2"** %b.addr, align 8
  store %"class.nvcuda::wmma::fragment.0"* %c, %"class.nvcuda::wmma::fragment.0"** %c.addr, align 8
  %0 = load %"class.nvcuda::wmma::fragment.0"*, %"class.nvcuda::wmma::fragment.0"** %d.addr, align 8
  %1 = bitcast %"class.nvcuda::wmma::fragment.0"* %0 to i32*
  %2 = load %"class.nvcuda::wmma::fragment"*, %"class.nvcuda::wmma::fragment"** %a.addr, align 8
  %3 = bitcast %"class.nvcuda::wmma::fragment"* %2 to i32*
  %4 = load %"class.nvcuda::wmma::fragment.2"*, %"class.nvcuda::wmma::fragment.2"** %b.addr, align 8
  %5 = bitcast %"class.nvcuda::wmma::fragment.2"* %4 to i32*
  %6 = load %"class.nvcuda::wmma::fragment.0"*, %"class.nvcuda::wmma::fragment.0"** %c.addr, align 8
  %7 = bitcast %"class.nvcuda::wmma::fragment.0"* %6 to i32*
  %8 = getelementptr i32, i32* %3, i32 0
  %9 = load i32, i32* %8, align 4
  %10 = bitcast i32 %9 to <2 x half>
  %11 = getelementptr i32, i32* %3, i32 1
  %12 = load i32, i32* %11, align 4
  %13 = bitcast i32 %12 to <2 x half>
  %14 = getelementptr i32, i32* %3, i32 2
  %15 = load i32, i32* %14, align 4
  %16 = bitcast i32 %15 to <2 x half>
  %17 = getelementptr i32, i32* %3, i32 3
  %18 = load i32, i32* %17, align 4
  %19 = bitcast i32 %18 to <2 x half>
  %20 = getelementptr i32, i32* %3, i32 4
  %21 = load i32, i32* %20, align 4
  %22 = bitcast i32 %21 to <2 x half>
  %23 = getelementptr i32, i32* %3, i32 5
  %24 = load i32, i32* %23, align 4
  %25 = bitcast i32 %24 to <2 x half>
  %26 = getelementptr i32, i32* %3, i32 6
  %27 = load i32, i32* %26, align 4
  %28 = bitcast i32 %27 to <2 x half>
  %29 = getelementptr i32, i32* %3, i32 7
  %30 = load i32, i32* %29, align 4
  %31 = bitcast i32 %30 to <2 x half>
  %32 = getelementptr i32, i32* %5, i32 0
  %33 = load i32, i32* %32, align 4
  %34 = bitcast i32 %33 to <2 x half>
  %35 = getelementptr i32, i32* %5, i32 1
  %36 = load i32, i32* %35, align 4
  %37 = bitcast i32 %36 to <2 x half>
  %38 = getelementptr i32, i32* %5, i32 2
  %39 = load i32, i32* %38, align 4
  %40 = bitcast i32 %39 to <2 x half>
  %41 = getelementptr i32, i32* %5, i32 3
  %42 = load i32, i32* %41, align 4
  %43 = bitcast i32 %42 to <2 x half>
  %44 = getelementptr i32, i32* %5, i32 4
  %45 = load i32, i32* %44, align 4
  %46 = bitcast i32 %45 to <2 x half>
  %47 = getelementptr i32, i32* %5, i32 5
  %48 = load i32, i32* %47, align 4
  %49 = bitcast i32 %48 to <2 x half>
  %50 = getelementptr i32, i32* %5, i32 6
  %51 = load i32, i32* %50, align 4
  %52 = bitcast i32 %51 to <2 x half>
  %53 = getelementptr i32, i32* %5, i32 7
  %54 = load i32, i32* %53, align 4
  %55 = bitcast i32 %54 to <2 x half>
  %56 = getelementptr i32, i32* %7, i32 0
  %57 = load i32, i32* %56, align 4
  %58 = bitcast i32 %57 to <2 x half>
  %59 = getelementptr i32, i32* %7, i32 1
  %60 = load i32, i32* %59, align 4
  %61 = bitcast i32 %60 to <2 x half>
  %62 = getelementptr i32, i32* %7, i32 2
  %63 = load i32, i32* %62, align 4
  %64 = bitcast i32 %63 to <2 x half>
  %65 = getelementptr i32, i32* %7, i32 3
  %66 = load i32, i32* %65, align 4
  %67 = bitcast i32 %66 to <2 x half>
  %68 = call { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.mma.row.col.f16.f16(<2 x half> %10, <2 x half> %13, <2 x half> %16, <2 x half> %19, <2 x half> %22, <2 x half> %25, <2 x half> %28, <2 x half> %31, <2 x half> %34, <2 x half> %37, <2 x half> %40, <2 x half> %43, <2 x half> %46, <2 x half> %49, <2 x half> %52, <2 x half> %55, <2 x half> %58, <2 x half> %61, <2 x half> %64, <2 x half> %67)
  %69 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half> } %68, 0
  %70 = bitcast <2 x half> %69 to i32
  %71 = getelementptr i32, i32* %1, i32 0
  store i32 %70, i32* %71, align 4
  %72 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half> } %68, 1
  %73 = bitcast <2 x half> %72 to i32
  %74 = getelementptr i32, i32* %1, i32 1
  store i32 %73, i32* %74, align 4
  %75 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half> } %68, 2
  %76 = bitcast <2 x half> %75 to i32
  %77 = getelementptr i32, i32* %1, i32 2
  store i32 %76, i32* %77, align 4
  %78 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half> } %68, 3
  %79 = bitcast <2 x half> %78 to i32
  %80 = getelementptr i32, i32* %1, i32 3
  store i32 %79, i32* %80, align 4
  ret void
}

; Function Attrs: convergent noinline nounwind optnone
define internal void @_ZN6nvcuda4wmmaL17store_matrix_syncEP6__halfRKNS0_8fragmentINS0_11accumulatorELi16ELi16ELi16ES1_vEEjNS0_8layout_tE(%struct.__half* %p, %"class.nvcuda::wmma::fragment.0"* nonnull align 4 dereferenceable(16) %a, i32 %ldm, i32 %layout) #0 {
entry:
  %p.addr = alloca %struct.__half*, align 8
  %a.addr = alloca %"class.nvcuda::wmma::fragment.0"*, align 8
  %ldm.addr = alloca i32, align 4
  %layout.addr = alloca i32, align 4
  store %struct.__half* %p, %struct.__half** %p.addr, align 8
  store %"class.nvcuda::wmma::fragment.0"* %a, %"class.nvcuda::wmma::fragment.0"** %a.addr, align 8
  store i32 %ldm, i32* %ldm.addr, align 4
  store i32 %layout, i32* %layout.addr, align 4
  %0 = load i32, i32* %layout.addr, align 4
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %1 = load %struct.__half*, %struct.__half** %p.addr, align 8
  %2 = bitcast %struct.__half* %1 to i32*
  %3 = load %"class.nvcuda::wmma::fragment.0"*, %"class.nvcuda::wmma::fragment.0"** %a.addr, align 8
  %4 = bitcast %"class.nvcuda::wmma::fragment.0"* %3 to i32*
  %5 = load i32, i32* %ldm.addr, align 4
  %6 = getelementptr i32, i32* %4, i32 0
  %7 = load i32, i32* %6, align 4
  %8 = bitcast i32 %7 to <2 x half>
  %9 = getelementptr i32, i32* %4, i32 1
  %10 = load i32, i32* %9, align 4
  %11 = bitcast i32 %10 to <2 x half>
  %12 = getelementptr i32, i32* %4, i32 2
  %13 = load i32, i32* %12, align 4
  %14 = bitcast i32 %13 to <2 x half>
  %15 = getelementptr i32, i32* %4, i32 3
  %16 = load i32, i32* %15, align 4
  %17 = bitcast i32 %16 to <2 x half>
  call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f16.p0i32(i32* %2, <2 x half> %8, <2 x half> %11, <2 x half> %14, <2 x half> %17, i32 %5)
  br label %if.end

if.else:                                          ; preds = %entry
  %18 = load %struct.__half*, %struct.__half** %p.addr, align 8
  %19 = bitcast %struct.__half* %18 to i32*
  %20 = load %"class.nvcuda::wmma::fragment.0"*, %"class.nvcuda::wmma::fragment.0"** %a.addr, align 8
  %21 = bitcast %"class.nvcuda::wmma::fragment.0"* %20 to i32*
  %22 = load i32, i32* %ldm.addr, align 4
  %23 = getelementptr i32, i32* %21, i32 0
  %24 = load i32, i32* %23, align 4
  %25 = bitcast i32 %24 to <2 x half>
  %26 = getelementptr i32, i32* %21, i32 1
  %27 = load i32, i32* %26, align 4
  %28 = bitcast i32 %27 to <2 x half>
  %29 = getelementptr i32, i32* %21, i32 2
  %30 = load i32, i32* %29, align 4
  %31 = bitcast i32 %30 to <2 x half>
  %32 = getelementptr i32, i32* %21, i32 3
  %33 = load i32, i32* %32, align 4
  %34 = bitcast i32 %33 to <2 x half>
  call void @llvm.nvvm.wmma.m16n16k16.store.d.col.stride.f16.p0i32(i32* %19, <2 x half> %25, <2 x half> %28, <2 x half> %31, <2 x half> %34, i32 %22)
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y() #2

; Function Attrs: convergent noinline nounwind optnone
define linkonce_odr dso_local void @_ZN6nvcuda4wmma8fragmentINS0_8matrix_aELi16ELi16ELi16E6__halfNS0_9row_majorEEC2Ev(%"class.nvcuda::wmma::fragment"* nonnull dereferenceable(32) %this) unnamed_addr #0 comdat align 2 {
entry:
  %this.addr = alloca %"class.nvcuda::wmma::fragment"*, align 8
  store %"class.nvcuda::wmma::fragment"* %this, %"class.nvcuda::wmma::fragment"** %this.addr, align 8
  %this1 = load %"class.nvcuda::wmma::fragment"*, %"class.nvcuda::wmma::fragment"** %this.addr, align 8
  %0 = bitcast %"class.nvcuda::wmma::fragment"* %this1 to %"struct.nvcuda::wmma::__frag_base"*
  call void @_ZN6nvcuda4wmma11__frag_baseI6__halfLi16ELi16EEC2Ev(%"struct.nvcuda::wmma::__frag_base"* nonnull dereferenceable(32) %0) #7
  ret void
}

; Function Attrs: convergent noinline nounwind optnone
define linkonce_odr dso_local void @_ZN6nvcuda4wmma11__frag_baseI6__halfLi16ELi16EEC2Ev(%"struct.nvcuda::wmma::__frag_base"* nonnull dereferenceable(32) %this) unnamed_addr #0 comdat align 2 {
entry:
  %this.addr = alloca %"struct.nvcuda::wmma::__frag_base"*, align 8
  store %"struct.nvcuda::wmma::__frag_base"* %this, %"struct.nvcuda::wmma::__frag_base"** %this.addr, align 8
  %this1 = load %"struct.nvcuda::wmma::__frag_base"*, %"struct.nvcuda::wmma::__frag_base"** %this.addr, align 8
  %x = getelementptr inbounds %"struct.nvcuda::wmma::__frag_base", %"struct.nvcuda::wmma::__frag_base"* %this1, i32 0, i32 0
  %array.begin = getelementptr inbounds [16 x %struct.__half], [16 x %struct.__half]* %x, i32 0, i32 0
  %arrayctor.end = getelementptr inbounds %struct.__half, %struct.__half* %array.begin, i64 16
  br label %arrayctor.loop

arrayctor.loop:                                   ; preds = %arrayctor.loop, %entry
  %arrayctor.cur = phi %struct.__half* [ %array.begin, %entry ], [ %arrayctor.next, %arrayctor.loop ]
  call void @_ZN6__halfC1Ev(%struct.__half* nonnull dereferenceable(2) %arrayctor.cur) #7
  %arrayctor.next = getelementptr inbounds %struct.__half, %struct.__half* %arrayctor.cur, i64 1
  %arrayctor.done = icmp eq %struct.__half* %arrayctor.next, %arrayctor.end
  br i1 %arrayctor.done, label %arrayctor.cont, label %arrayctor.loop

arrayctor.cont:                                   ; preds = %arrayctor.loop
  ret void
}

; Function Attrs: convergent noinline nounwind optnone
define linkonce_odr dso_local void @_ZN6__halfC1Ev(%struct.__half* nonnull dereferenceable(2) %this) unnamed_addr #0 comdat align 2 {
entry:
  %this.addr = alloca %struct.__half*, align 8
  store %struct.__half* %this, %struct.__half** %this.addr, align 8
  %this1 = load %struct.__half*, %struct.__half** %this.addr, align 8
  call void @_ZN6__halfC2Ev(%struct.__half* nonnull dereferenceable(2) %this1) #7
  ret void
}

; Function Attrs: convergent noinline nounwind optnone
define linkonce_odr dso_local void @_ZN6__halfC2Ev(%struct.__half* nonnull dereferenceable(2) %this) unnamed_addr #0 comdat align 2 {
entry:
  %this.addr = alloca %struct.__half*, align 8
  store %struct.__half* %this, %struct.__half** %this.addr, align 8
  %this1 = load %struct.__half*, %struct.__half** %this.addr, align 8
  ret void
}

; Function Attrs: convergent noinline nounwind optnone
define linkonce_odr dso_local void @_ZN6nvcuda4wmma8fragmentINS0_11accumulatorELi16ELi16ELi16E6__halfvEC2Ev(%"class.nvcuda::wmma::fragment.0"* nonnull dereferenceable(16) %this) unnamed_addr #0 comdat align 2 {
entry:
  %this.addr = alloca %"class.nvcuda::wmma::fragment.0"*, align 8
  store %"class.nvcuda::wmma::fragment.0"* %this, %"class.nvcuda::wmma::fragment.0"** %this.addr, align 8
  %this1 = load %"class.nvcuda::wmma::fragment.0"*, %"class.nvcuda::wmma::fragment.0"** %this.addr, align 8
  %0 = bitcast %"class.nvcuda::wmma::fragment.0"* %this1 to %"struct.nvcuda::wmma::__frag_base.1"*
  call void @_ZN6nvcuda4wmma11__frag_baseI6__halfLi8ELi8EEC2Ev(%"struct.nvcuda::wmma::__frag_base.1"* nonnull dereferenceable(16) %0) #7
  ret void
}

; Function Attrs: convergent noinline nounwind optnone
define linkonce_odr dso_local void @_ZN6nvcuda4wmma11__frag_baseI6__halfLi8ELi8EEC2Ev(%"struct.nvcuda::wmma::__frag_base.1"* nonnull dereferenceable(16) %this) unnamed_addr #0 comdat align 2 {
entry:
  %this.addr = alloca %"struct.nvcuda::wmma::__frag_base.1"*, align 8
  store %"struct.nvcuda::wmma::__frag_base.1"* %this, %"struct.nvcuda::wmma::__frag_base.1"** %this.addr, align 8
  %this1 = load %"struct.nvcuda::wmma::__frag_base.1"*, %"struct.nvcuda::wmma::__frag_base.1"** %this.addr, align 8
  %x = getelementptr inbounds %"struct.nvcuda::wmma::__frag_base.1", %"struct.nvcuda::wmma::__frag_base.1"* %this1, i32 0, i32 0
  %array.begin = getelementptr inbounds [8 x %struct.__half], [8 x %struct.__half]* %x, i32 0, i32 0
  %arrayctor.end = getelementptr inbounds %struct.__half, %struct.__half* %array.begin, i64 8
  br label %arrayctor.loop

arrayctor.loop:                                   ; preds = %arrayctor.loop, %entry
  %arrayctor.cur = phi %struct.__half* [ %array.begin, %entry ], [ %arrayctor.next, %arrayctor.loop ]
  call void @_ZN6__halfC1Ev(%struct.__half* nonnull dereferenceable(2) %arrayctor.cur) #7
  %arrayctor.next = getelementptr inbounds %struct.__half, %struct.__half* %arrayctor.cur, i64 1
  %arrayctor.done = icmp eq %struct.__half* %arrayctor.next, %arrayctor.end
  br i1 %arrayctor.done, label %arrayctor.cont, label %arrayctor.loop

arrayctor.cont:                                   ; preds = %arrayctor.loop
  ret void
}

; Function Attrs: convergent noinline nounwind optnone
define linkonce_odr dso_local void @_ZN6__halfC2Ef(%struct.__half* nonnull dereferenceable(2) %this, float %f) unnamed_addr #0 comdat align 2 {
entry:
  %this.addr = alloca %struct.__half*, align 8
  %f.addr = alloca float, align 4
  %agg.tmp = alloca %struct.__half, align 2
  store %struct.__half* %this, %struct.__half** %this.addr, align 8
  store float %f, float* %f.addr, align 4
  %this1 = load %struct.__half*, %struct.__half** %this.addr, align 8
  %0 = load float, float* %f.addr, align 4
  %call = call %struct.__half @_ZL12__float2halff(float %0) #7
  %1 = getelementptr inbounds %struct.__half, %struct.__half* %agg.tmp, i32 0, i32 0
  %2 = extractvalue %struct.__half %call, 0
  store i16 %2, i16* %1, align 2
  %__x = getelementptr inbounds %struct.__half, %struct.__half* %agg.tmp, i32 0, i32 0
  %3 = load i16, i16* %__x, align 2
  %__x2 = getelementptr inbounds %struct.__half, %struct.__half* %this1, i32 0, i32 0
  store i16 %3, i16* %__x2, align 2
  ret void
}

; Function Attrs: convergent noinline nounwind optnone
define internal %struct.__half @_ZL12__float2halff(float %a) #0 {
entry:
  %retval = alloca %struct.__half, align 2
  %a.addr = alloca float, align 4
  store float %a, float* %a.addr, align 4
  call void @_ZN6__halfC1Ev(%struct.__half* nonnull dereferenceable(2) %retval) #7
  %0 = bitcast %struct.__half* %retval to i16*
  %1 = load float, float* %a.addr, align 4
  %2 = call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %1) #8, !srcloc !14
  store i16 %2, i16* %0, align 2
  %3 = load %struct.__half, %struct.__half* %retval, align 2
  ret %struct.__half %3
}

; Function Attrs: argmemonly nounwind readonly
declare { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p0i32(i32* nocapture readonly, i32) #3

; Function Attrs: convergent noinline nounwind optnone
define linkonce_odr dso_local void @_ZN6nvcuda4wmma8fragmentINS0_8matrix_bELi16ELi16ELi16E6__halfNS0_9col_majorEEC2Ev(%"class.nvcuda::wmma::fragment.2"* nonnull dereferenceable(32) %this) unnamed_addr #0 comdat align 2 {
entry:
  %this.addr = alloca %"class.nvcuda::wmma::fragment.2"*, align 8
  store %"class.nvcuda::wmma::fragment.2"* %this, %"class.nvcuda::wmma::fragment.2"** %this.addr, align 8
  %this1 = load %"class.nvcuda::wmma::fragment.2"*, %"class.nvcuda::wmma::fragment.2"** %this.addr, align 8
  %0 = bitcast %"class.nvcuda::wmma::fragment.2"* %this1 to %"struct.nvcuda::wmma::__frag_base"*
  call void @_ZN6nvcuda4wmma11__frag_baseI6__halfLi16ELi16EEC2Ev(%"struct.nvcuda::wmma::__frag_base"* nonnull dereferenceable(32) %0) #7
  ret void
}

; Function Attrs: argmemonly nounwind readonly
declare { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.b.col.stride.f16.p0i32(i32* nocapture readonly, i32) #3

; Function Attrs: nounwind readnone
declare { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.mma.row.col.f16.f16(<2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>) #2

; Function Attrs: argmemonly nounwind writeonly
declare void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f16.p0i32(i32* nocapture writeonly, <2 x half>, <2 x half>, <2 x half>, <2 x half>, i32) #4

; Function Attrs: argmemonly nounwind writeonly
declare void @llvm.nvvm.wmma.m16n16k16.store.d.col.stride.f16.p0i32(i32* nocapture writeonly, <2 x half>, <2 x half>, <2 x half>, <2 x half>, i32) #4

; Function Attrs: convergent noinline nounwind optnone
define internal %struct.__half @_ZN6nvcuda4wmmaL19__get_storage_valueI6__halfS2_S2_EET0_T1_(%struct.__half* byval(%struct.__half) align 2 %in) #0 {
entry:
  %retval = alloca %struct.__half, align 2
  %0 = bitcast %struct.__half* %retval to i8*
  %1 = bitcast %struct.__half* %in to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 2 %0, i8* align 2 %1, i64 2, i1 false)
  %2 = load %struct.__half, %struct.__half* %retval, align 2
  ret %struct.__half %2
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #5

attributes #0 = { convergent noinline nounwind optnone "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_75" "target-features"="+ptx65,+sm_75" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent noinline norecurse nounwind optnone "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_75" "target-features"="+ptx65,+sm_75" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { argmemonly nounwind readonly }
attributes #4 = { argmemonly nounwind writeonly }
attributes #5 = { argmemonly nofree nosync nounwind willreturn }
attributes #6 = { nounwind }
attributes #7 = { convergent nounwind }
attributes #8 = { convergent nounwind readnone }

!llvm.module.flags = !{!0, !1, !2}
!nvvm.annotations = !{!3, !4, !5, !4, !6, !6, !6, !6, !7, !7, !6}
!llvm.ident = !{!8}
!nvvmir.version = !{!9}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 10, i32 2]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!3 = !{void ()* @_Z9test_wmmav, !"kernel", i32 1}
!4 = !{null, !"align", i32 8}
!5 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!6 = !{null, !"align", i32 16}
!7 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!8 = !{!"clang version 12.0.0 (https://github.com/llvm/llvm-project.git a89d751fb401540c89189e7c17ff64a6eca98587)"}
!9 = !{i32 1, i32 4}
!10 = !{i32 0, i32 1024}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.unroll.enable"}
!13 = distinct !{!13, !12}
!14 = !{i32 3803839}
