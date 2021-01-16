; ModuleID = 'wmma-sample.cu'
source_filename = "wmma-sample.cu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.dim3 = type { i32, i32, i32 }
%struct.CUstream_st = type opaque

$_ZN4dim3C2Ejjj = comdat any

; Function Attrs: noinline norecurse optnone uwtable
define dso_local void @_Z24__device_stub__test_wmmav() #0 {
entry:
  %grid_dim = alloca %struct.dim3, align 8
  %block_dim = alloca %struct.dim3, align 8
  %shmem_size = alloca i64, align 8
  %stream = alloca i8*, align 8
  %grid_dim.coerce = alloca { i64, i32 }, align 8
  %block_dim.coerce = alloca { i64, i32 }, align 8
  %kernel_args = alloca i8*, i64 1, align 16
  %0 = call i32 @__cudaPopCallConfiguration(%struct.dim3* %grid_dim, %struct.dim3* %block_dim, i64* %shmem_size, i8** %stream)
  %1 = load i64, i64* %shmem_size, align 8
  %2 = load i8*, i8** %stream, align 8
  %3 = bitcast { i64, i32 }* %grid_dim.coerce to i8*
  %4 = bitcast %struct.dim3* %grid_dim to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %3, i8* align 8 %4, i64 12, i1 false)
  %5 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %grid_dim.coerce, i32 0, i32 0
  %6 = load i64, i64* %5, align 8
  %7 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %grid_dim.coerce, i32 0, i32 1
  %8 = load i32, i32* %7, align 8
  %9 = bitcast { i64, i32 }* %block_dim.coerce to i8*
  %10 = bitcast %struct.dim3* %block_dim to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %9, i8* align 8 %10, i64 12, i1 false)
  %11 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %block_dim.coerce, i32 0, i32 0
  %12 = load i64, i64* %11, align 8
  %13 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %block_dim.coerce, i32 0, i32 1
  %14 = load i32, i32* %13, align 8
  %15 = bitcast i8* %2 to %struct.CUstream_st*
  %call = call i32 @cudaLaunchKernel(i8* bitcast (void ()* @_Z24__device_stub__test_wmmav to i8*), i64 %6, i32 %8, i64 %12, i32 %14, i8** %kernel_args, i64 %1, %struct.CUstream_st* %15)
  br label %setup.end

setup.end:                                        ; preds = %entry
  ret void
}

declare dso_local i32 @__cudaPopCallConfiguration(%struct.dim3*, %struct.dim3*, i64*, i8**)

declare dso_local i32 @cudaLaunchKernel(i8*, i64, i32, i64, i32, i8**, i64, %struct.CUstream_st*)

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #1

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z8TestWMMAv() #2 {
entry:
  %threads = alloca i32, align 4
  %blocks = alloca i32, align 4
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  store i32 256, i32* %threads, align 4
  store i32 10000, i32* %blocks, align 4
  %0 = load i32, i32* %blocks, align 4
  call void @_ZN4dim3C2Ejjj(%struct.dim3* nonnull dereferenceable(12) %agg.tmp, i32 %0, i32 1, i32 1)
  %1 = load i32, i32* %threads, align 4
  call void @_ZN4dim3C2Ejjj(%struct.dim3* nonnull dereferenceable(12) %agg.tmp1, i32 %1, i32 1, i32 1)
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*
  %3 = bitcast %struct.dim3* %agg.tmp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false)
  %4 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0
  %5 = load i64, i64* %4, align 4
  %6 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1
  %7 = load i32, i32* %6, align 4
  %8 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*
  %9 = bitcast %struct.dim3* %agg.tmp1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %8, i8* align 4 %9, i64 12, i1 false)
  %10 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 0
  %11 = load i64, i64* %10, align 4
  %12 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 1
  %13 = load i32, i32* %12, align 4
  %call = call i32 @__cudaPushCallConfiguration(i64 %5, i32 %7, i64 %11, i32 %13, i64 0, i8* null)
  %tobool = icmp ne i32 %call, 0
  br i1 %tobool, label %kcall.end, label %kcall.configok

kcall.configok:                                   ; preds = %entry
  call void @_Z24__device_stub__test_wmmav()
  br label %kcall.end

kcall.end:                                        ; preds = %kcall.configok, %entry
  ret void
}

declare dso_local i32 @__cudaPushCallConfiguration(i64, i32, i64, i32, i64, i8*) #3

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN4dim3C2Ejjj(%struct.dim3* nonnull dereferenceable(12) %this, i32 %vx, i32 %vy, i32 %vz) unnamed_addr #4 comdat align 2 {
entry:
  %this.addr = alloca %struct.dim3*, align 8
  %vx.addr = alloca i32, align 4
  %vy.addr = alloca i32, align 4
  %vz.addr = alloca i32, align 4
  store %struct.dim3* %this, %struct.dim3** %this.addr, align 8
  store i32 %vx, i32* %vx.addr, align 4
  store i32 %vy, i32* %vy.addr, align 4
  store i32 %vz, i32* %vz.addr, align 4
  %this1 = load %struct.dim3*, %struct.dim3** %this.addr, align 8
  %x = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 0
  %0 = load i32, i32* %vx.addr, align 4
  store i32 %0, i32* %x, align 4
  %y = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 1
  %1 = load i32, i32* %vy.addr, align 4
  store i32 %1, i32* %y, align 4
  %z = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 2
  %2 = load i32, i32* %vz.addr, align 4
  store i32 %2, i32* %z, align 4
  ret void
}

; Function Attrs: noinline norecurse optnone uwtable
define dso_local i32 @main() #0 {
entry:
  call void @_Z8TestWMMAv()
  %call = call i32 @cudaDeviceSynchronize()
  call void @_Z8TestWMMAv()
  %call1 = call i32 @cudaDeviceSynchronize()
  ret i32 0
}

declare dso_local i32 @cudaDeviceSynchronize() #3

attributes #0 = { noinline norecurse optnone uwtable "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nofree nosync nounwind willreturn }
attributes #2 = { noinline optnone uwtable "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { noinline nounwind optnone uwtable "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 10, i32 2]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{!"clang version 12.0.0 (https://github.com/llvm/llvm-project.git a89d751fb401540c89189e7c17ff64a6eca98587)"}
