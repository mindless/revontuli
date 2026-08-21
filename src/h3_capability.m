/* h3_capability.m
 *
 * I use this to answer, with real numbers rather than assumptions, whether the
 * RX 6900 XT can execute the operations h3.c actually needs. Everything here
 * runs on an EXPLICITLY selected MTLDevice — I never rely on
 * MTLCreateSystemDefaultDevice() picking the right GPU.
 *
 * The tests, in the order that matters for h3.c:
 *   1. compile h3_shaders.metal on the portable path (no H3_METAL_HAS_TENSOR)
 *   2. compile h3_shaders.metal on the M5 TensorOps path (expected to fail)
 *   3. compile a native MSL `bfloat` kernel (only the M5 path needs this)
 *   4. FP32 compute correctness + throughput
 *   5. FP16 (half) compute correctness
 *   6. emulated BF16-as-ushort compute correctness -- this is what h3.c's
 *      portable kernels actually do
 *   7. MPSGraph FP32 matmul vs CPU
 *   8. MPSGraph FP16 matmul vs CPU
 *   9. MPSGraph BF16 matmul vs CPU  <-- h3.c uses this on the portable path
 *  10. buffer allocation limits
 *
 * Build:
 *   xcrun clang -fobjc-arc -O2 -framework Foundation -framework Metal \
 *     -framework MetalPerformanceShaders \
 *     -framework MetalPerformanceShadersGraph \
 *     src/h3_capability.m -o bin/h3_capability
 *
 * Usage:
 *   ./bin/h3_capability [device-name-substring] [path/to/h3_shaders.metal]
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <math.h>
#include <string.h>

static int g_pass = 0, g_fail = 0;

static void result(const char *label, int ok, const char *detail) {
    printf("  [%s] %-46s %s\n", ok ? "PASS" : "FAIL", label,
           detail ? detail : "");
    if (ok) g_pass++; else g_fail++;
}

/* ---------- scalar conversion helpers (host side) ---------- */

static uint16_t f32_to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;                 /* round-to-nearest-even */
    return (uint16_t)(u >> 16);
}
static float bf16_to_f32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16; float f; memcpy(&f, &u, 4); return f;
}
static uint16_t f32_to_f16(float f) {
    _Float16 h = (_Float16)f; uint16_t b; memcpy(&b, &h, 2); return b;
}
static float f16_to_f32(uint16_t b) {
    _Float16 h; memcpy(&h, &b, 2); return (float)h;
}

/* ---------- inline MSL used by the dtype probes ---------- */

static NSString *const kProbeSource = @"\
#include <metal_stdlib>\n\
using namespace metal;\n\
\n\
/* FP32 fused multiply-add over a large array. */\n\
kernel void probe_fma_f32(device const float *a [[buffer(0)]],\n\
                          device const float *b [[buffer(1)]],\n\
                          device float *c [[buffer(2)]],\n\
                          uint i [[thread_position_in_grid]]) {\n\
    c[i] = fma(a[i], b[i], a[i]);\n\
}\n\
\n\
/* FP16 path. */\n\
kernel void probe_fma_f16(device const half *a [[buffer(0)]],\n\
                          device const half *b [[buffer(1)]],\n\
                          device half *c [[buffer(2)]],\n\
                          uint i [[thread_position_in_grid]]) {\n\
    c[i] = (half)fma((float)a[i], (float)b[i], (float)a[i]);\n\
}\n\
\n\
/* This mirrors exactly how h3.c's portable kernels handle BF16: the storage is\n\
   a plain ushort, the arithmetic happens in float, and conversion is done with\n\
   bit manipulation. No native `bfloat` type is involved. */\n\
static inline float h3_bf16_to_f32(ushort value) {\n\
    return as_type<float>((uint)value << 16);\n\
}\n\
static inline ushort h3_f32_to_bf16(float value) {\n\
    uint bits = as_type<uint>(value);\n\
    uint lsb = (bits >> 16) & 1u;\n\
    bits += 0x7fffu + lsb;\n\
    return (ushort)(bits >> 16);\n\
}\n\
kernel void probe_fma_bf16_emulated(device const ushort *a [[buffer(0)]],\n\
                                    device const ushort *b [[buffer(1)]],\n\
                                    device ushort *c [[buffer(2)]],\n\
                                    uint i [[thread_position_in_grid]]) {\n\
    float av = h3_bf16_to_f32(a[i]);\n\
    float bv = h3_bf16_to_f32(b[i]);\n\
    c[i] = h3_f32_to_bf16(fma(av, bv, av));\n\
}\n";

/* A separate tiny source: does this GPU's Metal compiler accept the NATIVE
 * bfloat type at all? Only the M5 TensorOps block needs this. */
static NSString *const kNativeBf16Source = @"\
#include <metal_stdlib>\n\
using namespace metal;\n\
kernel void probe_native_bfloat(device const bfloat *a [[buffer(0)]],\n\
                                device const bfloat *b [[buffer(1)]],\n\
                                device bfloat *c [[buffer(2)]],\n\
                                uint i [[thread_position_in_grid]]) {\n\
    c[i] = a[i] * b[i] + a[i];\n\
}\n";

/* ---------- device selection ---------- */

static id<MTLDevice> select_device(const char *wanted) {
    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
    NSString *needle = wanted ? [NSString stringWithUTF8String:wanted] : nil;
    if (needle.length) {
        for (id<MTLDevice> d in devices) {
            if ([d.name rangeOfString:needle
                              options:NSCaseInsensitiveSearch].location
                != NSNotFound) return d;
        }
        return nil;
    }
    return MTLCreateSystemDefaultDevice();
}

/* ---------- generic dispatch helper ---------- */

static int run_elementwise(id<MTLDevice> device, id<MTLCommandQueue> queue,
                           id<MTLLibrary> lib, NSString *fn,
                           id<MTLBuffer> a, id<MTLBuffer> b, id<MTLBuffer> c,
                           NSUInteger count, double *seconds) {
    NSError *err = nil;
    id<MTLFunction> f = [lib newFunctionWithName:fn];
    if (!f) { printf("      (no function %s)\n", fn.UTF8String); return 0; }
    id<MTLComputePipelineState> pipe =
        [device newComputePipelineStateWithFunction:f error:&err];
    if (!pipe) {
        printf("      (pipeline failed: %s)\n",
               err.localizedDescription.UTF8String);
        return 0;
    }
    NSUInteger tg = MIN((NSUInteger)256, pipe.maxTotalThreadsPerThreadgroup);
    NSDate *t0 = [NSDate date];
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:a offset:0 atIndex:0];
    [enc setBuffer:b offset:0 atIndex:1];
    [enc setBuffer:c offset:0 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (seconds) *seconds = -[t0 timeIntervalSinceNow];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        printf("      (command buffer status %ld: %s)\n", (long)cb.status,
               cb.error.localizedDescription.UTF8String);
        return 0;
    }
    return 1;
}

/* ---------- MPSGraph matmul vs CPU ---------- */

static int mpsgraph_matmul(id<MTLDevice> device, id<MTLCommandQueue> queue,
                           MPSDataType dtype, const char *label,
                           NSUInteger M, NSUInteger K, NSUInteger N,
                           double tolerance) {
    @autoreleasepool {
        size_t esz = (dtype == MPSDataTypeFloat32) ? 4 : 2;
        id<MTLBuffer> ab = [device newBufferWithLength:M * K * esz
                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> bb = [device newBufferWithLength:K * N * esz
                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> cb = [device newBufferWithLength:M * N * esz
                                              options:MTLResourceStorageModeShared];
        if (!ab || !bb || !cb) { result(label, 0, "buffer allocation failed"); return 0; }

        float *hostA = malloc(M * K * sizeof(float));
        float *hostB = malloc(K * N * sizeof(float));
        /* Small, well-conditioned values so that low-precision formats are
         * still comparable against the CPU reference. */
        for (NSUInteger i = 0; i < M * K; i++)
            hostA[i] = (float)(((int)(i * 37u % 17u)) - 8) * 0.0625f;
        for (NSUInteger i = 0; i < K * N; i++)
            hostB[i] = (float)(((int)(i * 53u % 13u)) - 6) * 0.0625f;

        /* Quantise the inputs to the target dtype, then use the DEQUANTISED
         * values for the CPU reference. That way I measure the GPU's matmul
         * error, not the input rounding error. */
        for (NSUInteger i = 0; i < M * K; i++) {
            if (dtype == MPSDataTypeFloat32) ((float *)ab.contents)[i] = hostA[i];
            else if (dtype == MPSDataTypeFloat16) {
                uint16_t q = f32_to_f16(hostA[i]);
                ((uint16_t *)ab.contents)[i] = q; hostA[i] = f16_to_f32(q);
            } else {
                uint16_t q = f32_to_bf16(hostA[i]);
                ((uint16_t *)ab.contents)[i] = q; hostA[i] = bf16_to_f32(q);
            }
        }
        for (NSUInteger i = 0; i < K * N; i++) {
            if (dtype == MPSDataTypeFloat32) ((float *)bb.contents)[i] = hostB[i];
            else if (dtype == MPSDataTypeFloat16) {
                uint16_t q = f32_to_f16(hostB[i]);
                ((uint16_t *)bb.contents)[i] = q; hostB[i] = f16_to_f32(q);
            } else {
                uint16_t q = f32_to_bf16(hostB[i]);
                ((uint16_t *)bb.contents)[i] = q; hostB[i] = bf16_to_f32(q);
            }
        }

        MPSGraph *graph = [[MPSGraph alloc] init];
        NSArray<NSNumber *> *shapeA = @[@(M), @(K)];
        NSArray<NSNumber *> *shapeB = @[@(K), @(N)];
        MPSGraphTensor *ta = [graph placeholderWithShape:shapeA dataType:dtype name:nil];
        MPSGraphTensor *tb = [graph placeholderWithShape:shapeB dataType:dtype name:nil];
        MPSGraphTensor *tc = [graph matrixMultiplicationWithPrimaryTensor:ta
                                                         secondaryTensor:tb
                                                                    name:nil];
        MPSGraphTensorData *da = [[MPSGraphTensorData alloc] initWithMTLBuffer:ab
                                                                        shape:shapeA
                                                                     dataType:dtype];
        MPSGraphTensorData *db = [[MPSGraphTensorData alloc] initWithMTLBuffer:bb
                                                                        shape:shapeB
                                                                     dataType:dtype];
        MPSGraphTensorData *dc = [[MPSGraphTensorData alloc] initWithMTLBuffer:cb
                                                                        shape:@[@(M), @(N)]
                                                                     dataType:dtype];
        if (!da || !db || !dc) {
            result(label, 0, "MPSGraphTensorData creation failed");
            free(hostA); free(hostB); return 0;
        }

        NSDate *t0 = [NSDate date];
        @try {
            [graph runWithMTLCommandQueue:queue
                                    feeds:@{ ta: da, tb: db }
                         targetOperations:nil
                        resultsDictionary:@{ tc: dc }];
        } @catch (NSException *ex) {
            char detail[256];
            snprintf(detail, sizeof(detail), "exception: %s",
                     ex.reason.UTF8String ? ex.reason.UTF8String : "unknown");
            result(label, 0, detail);
            free(hostA); free(hostB); return 0;
        }
        double secs = -[t0 timeIntervalSinceNow];

        /* CPU reference on a sampled set of output elements. */
        double worst = 0.0; NSUInteger nan_count = 0, checked = 0;
        NSUInteger step = (M * N > 4096) ? (M * N) / 4096 : 1;
        for (NSUInteger idx = 0; idx < M * N; idx += step) {
            NSUInteger r = idx / N, c = idx % N;
            double ref = 0.0;
            for (NSUInteger k = 0; k < K; k++)
                ref += (double)hostA[r * K + k] * (double)hostB[k * N + c];
            float got;
            if (dtype == MPSDataTypeFloat32)      got = ((float *)cb.contents)[idx];
            else if (dtype == MPSDataTypeFloat16) got = f16_to_f32(((uint16_t *)cb.contents)[idx]);
            else                                  got = bf16_to_f32(((uint16_t *)cb.contents)[idx]);
            if (isnan(got) || isinf(got)) { nan_count++; continue; }
            double denom = fabs(ref) > 1.0 ? fabs(ref) : 1.0;
            double rel = fabs((double)got - ref) / denom;
            if (rel > worst) worst = rel;
            checked++;
        }

        double gflops = 2.0 * (double)M * (double)K * (double)N / secs / 1e9;
        char detail[320];
        snprintf(detail, sizeof(detail),
                 "%lux%lux%lu  checked=%lu  NaN/Inf=%lu  worst_rel=%.3g  "
                 "%.3f s  %.1f GFLOP/s",
                 (unsigned long)M, (unsigned long)K, (unsigned long)N,
                 (unsigned long)checked, (unsigned long)nan_count, worst,
                 secs, gflops);
        int ok = (nan_count == 0) && (checked > 0) && (worst <= tolerance);
        result(label, ok, detail);
        free(hostA); free(hostB);
        return ok;
    }
}

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *wanted = (argc > 1) ? argv[1] : "6900";
        const char *shaderPath = (argc > 2) ? argv[2] :
            "src/h3.c/h3_shaders.metal";

        id<MTLDevice> device = select_device(wanted);
        if (!device) {
            printf("I could not find a Metal device matching \"%s\".\n", wanted);
            return 2;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        printf("I selected Metal GPU:\n");
        printf("  name                  = %s\n", device.name.UTF8String);
        printf("  registryID            = 0x%llx\n",
               (unsigned long long)device.registryID);
        if (@available(macOS 14.0, *))
            printf("  architecture          = %s\n", device.architecture.name.UTF8String);
        printf("  unifiedMemory         = %s\n", device.hasUnifiedMemory ? "yes" : "no");
        printf("  removable             = %s\n", device.isRemovable ? "yes" : "no");
        printf("  recommendedWorkingSet = %.2f GiB\n",
               (double)device.recommendedMaxWorkingSetSize / 1073741824.0);
        printf("  maxBufferLength       = %.2f GiB\n",
               (double)device.maxBufferLength / 1073741824.0);
        printf("\n");

        /* ---------- 1 & 2: the real h3.c shader set ---------- */
        printf("=== h3.c shader compilation on this GPU ===\n");
        NSError *err = nil;
        NSString *src = [NSString stringWithContentsOfFile:
                            [NSString stringWithUTF8String:shaderPath]
                                                 encoding:NSUTF8StringEncoding
                                                    error:&err];
        if (!src) {
            printf("  I could not read %s: %s\n", shaderPath,
                   err.localizedDescription.UTF8String);
        } else {
            /* Portable path: exactly what h3_gpu.m compiles for a non-M5 GPU. */
            MTLCompileOptions *portable = [[MTLCompileOptions alloc] init];
            portable.mathMode = MTLMathModeSafe;
            err = nil;
            NSDate *t0 = [NSDate date];
            id<MTLLibrary> plib = [device newLibraryWithSource:src
                                                       options:portable
                                                         error:&err];
            double psecs = -[t0 timeIntervalSinceNow];
            char detail[256];
            snprintf(detail, sizeof(detail), "%.1f s, %lu functions", psecs,
                     (unsigned long)plib.functionNames.count);
            result("h3_shaders.metal PORTABLE (no TensorOps)", plib != nil,
                   plib ? detail : "see error below");
            if (!plib)
                printf("      ERROR: %s\n", err.localizedDescription.UTF8String);

            /* M5 path: I expect this to fail here, which is fine, because
             * h3_gpu.m only enables it when the device name contains "M5". */
            MTLCompileOptions *m5 = [[MTLCompileOptions alloc] init];
            m5.mathMode = MTLMathModeSafe;
            m5.preprocessorMacros = @{ @"H3_METAL_HAS_TENSOR": @"1" };
            err = nil;
            id<MTLLibrary> mlib = [device newLibraryWithSource:src
                                                      options:m5
                                                        error:&err];
            printf("  [%s] %-46s %s\n", mlib ? "PASS" : "n/a",
                   "h3_shaders.metal M5 TENSOROPS path",
                   mlib ? "compiled" : "not available (expected on AMD)");
            if (!mlib) {
                NSString *d = err.localizedDescription ?: @"";
                if (d.length > 400) d = [d substringToIndex:400];
                printf("      first error: %s\n", d.UTF8String);
            }
        }
        printf("\n");

        /* ---------- 3: native bfloat type ---------- */
        printf("=== Metal shading-language dtype support ===\n");
        err = nil;
        id<MTLLibrary> nativeBf = [device newLibraryWithSource:kNativeBf16Source
                                                      options:nil error:&err];
        result("MSL native `bfloat` type compiles", nativeBf != nil,
               nativeBf ? "" : "not supported by this Metal compiler/GPU");
        if (!nativeBf) {
            NSString *d = err.localizedDescription ?: @"";
            if (d.length > 240) d = [d substringToIndex:240];
            printf("      %s\n", d.UTF8String);
        }

        err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:kProbeSource
                                                 options:nil error:&err];
        if (!lib) {
            printf("  I could not compile the probe kernels: %s\n",
                   err.localizedDescription.UTF8String);
            return 3;
        }

        /* ---------- 4-6: elementwise correctness in each dtype ---------- */
        const NSUInteger COUNT = 16u * 1024u * 1024u;   /* 16 Mi elements */

        /* FP32 */
        {
            id<MTLBuffer> a = [device newBufferWithLength:COUNT * 4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> b = [device newBufferWithLength:COUNT * 4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> c = [device newBufferWithLength:COUNT * 4 options:MTLResourceStorageModeShared];
            float *pa = a.contents, *pb = b.contents;
            for (NSUInteger i = 0; i < COUNT; i++) {
                pa[i] = (float)((long)(i % 1000) - 500) * 0.001f;
                pb[i] = (float)((long)(i % 777) - 388) * 0.002f;
            }
            double secs = 0;
            int ran = run_elementwise(device, queue, lib, @"probe_fma_f32", a, b, c, COUNT, &secs);
            double worst = 0; NSUInteger bad = 0;
            if (ran) {
                float *pc = c.contents;
                for (NSUInteger i = 0; i < COUNT; i += 997) {
                    float ref = fmaf(pa[i], pb[i], pa[i]);
                    if (isnan(pc[i])) { bad++; continue; }
                    double d = fabs((double)pc[i] - (double)ref);
                    if (d > worst) worst = d;
                }
            }
            char detail[256];
            snprintf(detail, sizeof(detail),
                     "%lu Mi elems  NaN=%lu  max_abs_err=%.3g  %.4f s  %.2f GB/s",
                     (unsigned long)(COUNT >> 20), (unsigned long)bad, worst, secs,
                     (double)COUNT * 12.0 / secs / 1e9);
            result("Metal FP32 elementwise FMA", ran && bad == 0 && worst < 1e-6, detail);
        }

        /* FP16 */
        {
            id<MTLBuffer> a = [device newBufferWithLength:COUNT * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> b = [device newBufferWithLength:COUNT * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> c = [device newBufferWithLength:COUNT * 2 options:MTLResourceStorageModeShared];
            uint16_t *pa = a.contents, *pb = b.contents;
            for (NSUInteger i = 0; i < COUNT; i++) {
                pa[i] = f32_to_f16((float)((long)(i % 1000) - 500) * 0.001f);
                pb[i] = f32_to_f16((float)((long)(i % 777) - 388) * 0.002f);
            }
            double secs = 0;
            int ran = run_elementwise(device, queue, lib, @"probe_fma_f16", a, b, c, COUNT, &secs);
            double worst = 0; NSUInteger bad = 0;
            if (ran) {
                uint16_t *pc = c.contents;
                for (NSUInteger i = 0; i < COUNT; i += 997) {
                    float av = f16_to_f32(pa[i]), bv = f16_to_f32(pb[i]);
                    float ref = fmaf(av, bv, av);
                    float got = f16_to_f32(pc[i]);
                    if (isnan(got)) { bad++; continue; }
                    double d = fabs((double)got - (double)ref);
                    if (d > worst) worst = d;
                }
            }
            char detail[256];
            snprintf(detail, sizeof(detail),
                     "%lu Mi elems  NaN=%lu  max_abs_err=%.3g  %.4f s",
                     (unsigned long)(COUNT >> 20), (unsigned long)bad, worst, secs);
            result("Metal FP16 (half) elementwise FMA", ran && bad == 0 && worst < 1e-2, detail);
        }

        /* Emulated BF16 -- h3.c's actual portable representation */
        {
            id<MTLBuffer> a = [device newBufferWithLength:COUNT * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> b = [device newBufferWithLength:COUNT * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> c = [device newBufferWithLength:COUNT * 2 options:MTLResourceStorageModeShared];
            uint16_t *pa = a.contents, *pb = b.contents;
            for (NSUInteger i = 0; i < COUNT; i++) {
                pa[i] = f32_to_bf16((float)((long)(i % 1000) - 500) * 0.001f);
                pb[i] = f32_to_bf16((float)((long)(i % 777) - 388) * 0.002f);
            }
            double secs = 0;
            int ran = run_elementwise(device, queue, lib, @"probe_fma_bf16_emulated",
                                      a, b, c, COUNT, &secs);
            double worst = 0; NSUInteger bad = 0, mismatch = 0;
            if (ran) {
                uint16_t *pc = c.contents;
                for (NSUInteger i = 0; i < COUNT; i += 997) {
                    float av = bf16_to_f32(pa[i]), bv = bf16_to_f32(pb[i]);
                    uint16_t refbits = f32_to_bf16(fmaf(av, bv, av));
                    float got = bf16_to_f32(pc[i]);
                    if (isnan(got)) { bad++; continue; }
                    /* I require BIT-EXACT agreement with the host converter,
                     * because h3.c's CPU and GPU sides must round identically. */
                    if (pc[i] != refbits) mismatch++;
                    double d = fabs((double)got - (double)bf16_to_f32(refbits));
                    if (d > worst) worst = d;
                }
            }
            char detail[256];
            snprintf(detail, sizeof(detail),
                     "%lu Mi elems  NaN=%lu  bit_mismatch=%lu  %.4f s",
                     (unsigned long)(COUNT >> 20), (unsigned long)bad,
                     (unsigned long)mismatch, secs);
            result("Metal BF16-as-ushort (h3.c portable style)",
                   ran && bad == 0 && mismatch == 0, detail);
        }
        printf("\n");

        /* ---------- 7-9: MPSGraph ---------- */
        printf("=== MPSGraph matrix multiplication vs CPU reference ===\n");
        mpsgraph_matmul(device, queue, MPSDataTypeFloat32,  "MPSGraph FP32 matmul", 512, 512, 512, 1e-4);
        mpsgraph_matmul(device, queue, MPSDataTypeFloat16,  "MPSGraph FP16 matmul", 512, 512, 512, 5e-2);
        mpsgraph_matmul(device, queue, MPSDataTypeBFloat16, "MPSGraph BF16 matmul", 512, 512, 512, 1e-1);
        /* A second, larger BF16 shape, closer to H3's real DiT dimensions. */
        mpsgraph_matmul(device, queue, MPSDataTypeBFloat16, "MPSGraph BF16 matmul (large)", 2048, 3072, 3072, 1e-1);
        printf("\n");

        /* ---------- 10: allocation limits ---------- */
        printf("=== Buffer allocation limits on this GPU ===\n");
        NSMutableArray *held = [NSMutableArray array];
        double total = 0;
        for (int i = 0; i < 64; i++) {
            id<MTLBuffer> buf = [device newBufferWithLength:512ull * 1024 * 1024
                                                   options:MTLResourceStorageModePrivate];
            if (!buf) break;
            [held addObject:buf];
            total += 0.5;
        }
        char detail[192];
        snprintf(detail, sizeof(detail),
                 "allocated %.1f GiB in 512 MiB private chunks; live=%.2f GiB",
                 total, (double)device.currentAllocatedSize / 1073741824.0);
        result("Private (VRAM) allocation reaches >= 8 GiB", total >= 8.0, detail);
        [held removeAllObjects];

        /* Largest single buffer -- this matters because AMD reports a much
         * smaller maxBufferLength than Apple Silicon does. */
        {
            uint64_t lo = 0, hi = device.maxBufferLength, best = 0;
            for (int i = 0; i < 24 && lo <= hi; i++) {
                uint64_t mid = lo + (hi - lo) / 2;
                if (mid < 1024) break;
                id<MTLBuffer> buf = [device newBufferWithLength:mid
                                                       options:MTLResourceStorageModePrivate];
                if (buf) { best = mid; lo = mid + (1ull << 20); }
                else     { hi = mid - (1ull << 20); }
            }
            snprintf(detail, sizeof(detail),
                     "largest single MTLBuffer = %.2f GiB (reported cap %.2f GiB)",
                     (double)best / 1073741824.0,
                     (double)device.maxBufferLength / 1073741824.0);
            result("Single-buffer allocation matches reported cap",
                   best > (uint64_t)(0.9 * (double)device.maxBufferLength), detail);
        }

        printf("\n================ SUMMARY ================\n");
        printf("I ran %d checks on \"%s\": %d passed, %d failed.\n",
               g_pass + g_fail, device.name.UTF8String, g_pass, g_fail);
        return g_fail ? 1 : 0;
    }
}
