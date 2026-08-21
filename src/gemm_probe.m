/* gemm_probe.m — measure GEMM variants on the eGPU at H3 DiT shapes.
 *
 * The DiT denoise runs 800 MPSGraph BF16 matmuls per 4-step generation
 * (qkv/out/fc1/fc2 x 50 blocks x 4 steps) and they dominate GPU time.
 * This probe measures, in steady state, what each available backend
 * sustains at those exact shapes so we know the ceiling before touching
 * h3_gpu.m.
 *
 * Variants:
 *   graph-bf16-T   MPSGraph BF16, weight [N,K] transposed in-graph (production)
 *   graph-bf16     MPSGraph BF16, weight pre-transposed [K,N]
 *   graph-f16-T    MPSGraph FP16, in-graph transpose
 *   graph-f16      MPSGraph FP16, pre-transposed
 *   graph-f32-T    MPSGraph FP32, in-graph transpose
 *   mps-f16        MPSMatrixMultiplication FP16, transposeRight (weight [N,K])
 *   mps-f32        MPSMatrixMultiplication FP32, transposeRight
 *   tile-bf16      h3's portable 16x16 BF16 tile shader (h3_linear_bf16)
 *
 * Build:
 *   xcrun clang -fobjc-arc -O2 -framework Foundation -framework Metal \
 *     -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
 *     src/gemm_probe.m -o bin/gemm_probe
 * Run:
 *   ./bin/gemm_probe "6900" [rows]        (default rows 7488 = 39f @640x832)
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <mach/mach_time.h>

static double now_seconds(void) {
    static mach_timebase_info_data_t info;
    if (!info.denom) mach_timebase_info(&info);
    return (double)mach_absolute_time() * info.numer / info.denom / 1e9;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t rounding = ((bits >> 16) & 1u) + 0x7fffu;
    return (uint16_t)((bits + rounding) >> 16);
}

static float bf16_to_f32(uint16_t bits) {
    uint32_t wide = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &wide, sizeof(value));
    return value;
}

/* The same portable 16x16 BF16 tile kernel h3_shaders.metal uses as its
 * non-MPS fallback, minus bias, so tile-bf16 is measured faithfully. */
static NSString *const kTileSource = @"\n"
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct linear_args { uint rows, input_dim, output_dim, has_bias; };\n"
"inline float bf16_load(const device ushort *p, uint i) {\n"
"    return as_type<float>((uint)p[i] << 16);\n"
"}\n"
"kernel void h3_linear_bf16(const device ushort *input [[buffer(0)]],\n"
"                           const device ushort *weight [[buffer(1)]],\n"
"                           const device ushort *bias [[buffer(2)]],\n"
"                           device ushort *output [[buffer(3)]],\n"
"                           constant linear_args &args [[buffer(4)]],\n"
"                           uint2 gid [[thread_position_in_grid]],\n"
"                           uint2 lid [[thread_position_in_threadgroup]]) {\n"
"    threadgroup float a_tile[16][16];\n"
"    threadgroup float b_tile[16][16];\n"
"    uint row = gid.y, column = gid.x;\n"
"    float sum = 0.0f;\n"
"    uint tiles = (args.input_dim + 15) / 16;\n"
"    for (uint t = 0; t < tiles; t++) {\n"
"        uint k_a = t * 16 + lid.x;\n"
"        uint k_b = t * 16 + lid.y;\n"
"        a_tile[lid.y][lid.x] = (row < args.rows && k_a < args.input_dim) ?\n"
"            bf16_load(input, row * args.input_dim + k_a) : 0.0f;\n"
"        b_tile[lid.y][lid.x] = (column < args.output_dim && k_b < args.input_dim) ?\n"
"            bf16_load(weight, column * args.input_dim + k_b) : 0.0f;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        for (uint k = 0; k < 16; k++) sum += a_tile[lid.y][k] * b_tile[k][lid.x];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (row < args.rows && column < args.output_dim) {\n"
"        uint bits = as_type<uint>(sum);\n"
"        uint rounded = (bits + (((bits >> 16) & 1u) + 0x7fffu)) >> 16;\n"
"        output[row * args.output_dim + column] = (ushort)rounded;\n"
"    }\n"
"}\n";

typedef struct {
    const char *name;
    uint32_t k, n;
} gemm_shape;

typedef struct {
    id<MTLBuffer> input;      /* rows x K, element type per variant */
    id<MTLBuffer> weightNK;   /* N x K row-major (production layout) */
    id<MTLBuffer> weightKN;   /* K x N row-major (pre-transposed) */
    id<MTLBuffer> output;     /* rows x N */
} gemm_buffers;

static id<MTLBuffer> private_buffer_with(id<MTLDevice> device,
                                         id<MTLCommandQueue> queue,
                                         const void *bytes, size_t length) {
    id<MTLBuffer> staging = [device newBufferWithBytes:bytes length:length
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> target = [device newBufferWithLength:length
        options:MTLResourceStorageModePrivate];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:target
        destinationOffset:0 size:length];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    return target;
}

/* One CPU reference dot product to catch a transposed or garbage result. */
static float cpu_dot(const float *a, const float *w_nk, uint32_t k,
                     uint32_t row_a, uint32_t col_n, uint32_t lda) {
    double sum = 0.0;
    for (uint32_t i = 0; i < k; i++)
        sum += (double)a[row_a * lda + i] * (double)w_nk[col_n * k + i];
    return (float)sum;
}

typedef double (*run_fn)(void);

static double time_variant(id<MTLCommandQueue> queue,
                           void (^encode)(id<MTLCommandBuffer>),
                           int warmups, int iterations) {
    for (int i = 0; i < warmups; i++) {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        encode(command);
        [command commit];
        [command waitUntilCompleted];
        if (command.status == MTLCommandBufferStatusError) return -1.0;
    }
    double gpu = 0.0;
    double wall_start = now_seconds();
    for (int i = 0; i < iterations; i++) {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        encode(command);
        [command commit];
        [command waitUntilCompleted];
        if (command.status == MTLCommandBufferStatusError) return -1.0;
        if (command.GPUEndTime > command.GPUStartTime)
            gpu += command.GPUEndTime - command.GPUStartTime;
    }
    double wall = now_seconds() - wall_start;
    /* MPSGraph may retire work via child buffers the root cannot see, so use
     * wall time when the root's GPU time is implausibly small. */
    return gpu > wall * 0.25 ? gpu / iterations : wall / iterations;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *want = argc > 1 ? argv[1] : "6900";
        uint32_t rows = argc > 2 ? (uint32_t)atoi(argv[2]) : 7488;
        id<MTLDevice> device = nil;
        for (id<MTLDevice> candidate in MTLCopyAllDevices()) {
            if ([candidate.name rangeOfString:@(want)].location != NSNotFound) {
                device = candidate;
                break;
            }
        }
        if (!device) {
            fprintf(stderr, "no Metal device matching '%s'\n", want);
            return 1;
        }
        printf("device: %s | rows=%u\n", device.name.UTF8String, rows);
        id<MTLCommandQueue> queue = [device newCommandQueue];

        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:kTileSource
            options:nil error:&error];
        id<MTLComputePipelineState> tile = nil;
        if (library) {
            tile = [device newComputePipelineStateWithFunction:
                [library newFunctionWithName:@"h3_linear_bf16"] error:&error];
        }
        if (!tile) fprintf(stderr, "tile shader unavailable: %s\n",
                           error.localizedDescription.UTF8String);

        gemm_shape shapes[] = {
            {"qkv 5376->21504", 5376, 21504},
            {"out 7168->5376",  7168, 5376},
            {"fc1 5376->28672", 5376, 28672},
            {"fc2 14336->5376", 14336, 5376},
        };
        double totals[16] = {0};
        const char *names[16] = {0};
        int variant_count = 0;

        for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
            uint32_t K = shapes[s].k, N = shapes[s].n;
            double flops = 2.0 * rows * K * N;
            printf("\n== %s (%.1f GFLOP/dispatch) ==\n", shapes[s].name,
                   flops / 1e9);

            size_t a_elems = (size_t)rows * K;
            size_t w_elems = (size_t)N * K;
            size_t o_elems = (size_t)rows * N;
            float *a_f32 = malloc(a_elems * sizeof(float));
            float *w_f32 = malloc(w_elems * sizeof(float));
            uint16_t *a_bf16 = malloc(a_elems * 2);
            uint16_t *w_bf16_nk = malloc(w_elems * 2);
            uint16_t *w_bf16_kn = malloc(w_elems * 2);
            __fp16 *a_f16 = malloc(a_elems * 2);
            __fp16 *w_f16_nk = malloc(w_elems * 2);
            __fp16 *w_f16_kn = malloc(w_elems * 2);
            srand(42 + (int)s);
            for (size_t i = 0; i < a_elems; i++) {
                a_f32[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
                a_bf16[i] = f32_to_bf16(a_f32[i]);
                a_f32[i] = bf16_to_f32(a_bf16[i]);
                a_f16[i] = (__fp16)a_f32[i];
            }
            for (size_t i = 0; i < w_elems; i++) {
                w_f32[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.06f;
                w_bf16_nk[i] = f32_to_bf16(w_f32[i]);
                w_f32[i] = bf16_to_f32(w_bf16_nk[i]);
                w_f16_nk[i] = (__fp16)w_f32[i];
            }
            for (uint32_t nn = 0; nn < N; nn++)
                for (uint32_t kk = 0; kk < K; kk++) {
                    w_bf16_kn[(size_t)kk * N + nn] = w_bf16_nk[(size_t)nn * K + kk];
                    w_f16_kn[(size_t)kk * N + nn] = w_f16_nk[(size_t)nn * K + kk];
                }
            float reference = cpu_dot(a_f32, w_f32, K, 3, 5, K);

            id<MTLBuffer> in_bf16 = private_buffer_with(device, queue, a_bf16, a_elems * 2);
            id<MTLBuffer> w_nk_bf16 = private_buffer_with(device, queue, w_bf16_nk, w_elems * 2);
            id<MTLBuffer> w_kn_bf16 = private_buffer_with(device, queue, w_bf16_kn, w_elems * 2);
            id<MTLBuffer> in_f16 = private_buffer_with(device, queue, a_f16, a_elems * 2);
            id<MTLBuffer> w_nk_f16 = private_buffer_with(device, queue, w_f16_nk, w_elems * 2);
            __fp16 *w_f16_scaled = malloc(w_elems * 2);
            for (size_t i = 0; i < w_elems; i++)
                w_f16_scaled[i] = (__fp16)((float)w_f16_nk[i] * 0.125f);
            id<MTLBuffer> w_nk_f16s = private_buffer_with(device, queue,
                w_f16_scaled, w_elems * 2);
            free(w_f16_scaled);
            id<MTLBuffer> w_kn_f16 = private_buffer_with(device, queue, w_f16_kn, w_elems * 2);
            id<MTLBuffer> in_f32 = private_buffer_with(device, queue, a_f32, a_elems * 4);
            id<MTLBuffer> w_nk_f32 = private_buffer_with(device, queue, w_f32, w_elems * 4);
            id<MTLBuffer> out16 = [device newBufferWithLength:o_elems * 2
                options:MTLResourceStorageModePrivate];
            id<MTLBuffer> out32 = [device newBufferWithLength:o_elems * 4
                options:MTLResourceStorageModePrivate];
            id<MTLBuffer> check = [device newBufferWithLength:16
                options:MTLResourceStorageModeShared];
            free(a_f32); free(w_f32); free(a_bf16); free(w_bf16_nk);
            free(w_bf16_kn); free(a_f16); free(w_f16_nk); free(w_f16_kn);

            /* --- MPSGraph variants --- */
            struct { const char *label; MPSDataType type; int transposed; } gvs[] = {
                {"graph-bf16-T", MPSDataTypeBFloat16, 1},
                {"graph-bf16",   MPSDataTypeBFloat16, 0},
                {"graph-f16-T",  MPSDataTypeFloat16, 1},
                {"graph-f16",    MPSDataTypeFloat16, 0},
                {"graph-f32-T",  MPSDataTypeFloat32, 1},
            };
            int vi = 0;
            for (size_t g = 0; g < sizeof(gvs) / sizeof(gvs[0]); g++) {
                MPSDataType type = gvs[g].type;
                int transposed = gvs[g].transposed;
                MPSGraph *graph = [[MPSGraph alloc] init];
                NSArray *inShape = @[@1, @(rows), @(K)];
                NSArray *wShape = transposed ? @[@1, @(N), @(K)] : @[@1, @(K), @(N)];
                NSArray *outShape = @[@1, @(rows), @(N)];
                MPSGraphTensor *tin = [graph placeholderWithShape:inShape
                    dataType:type name:nil];
                MPSGraphTensor *tw = [graph placeholderWithShape:wShape
                    dataType:type name:nil];
                MPSGraphTensor *rhs = transposed ?
                    [graph transposeTensor:tw dimension:1 withDimension:2 name:nil] : tw;
                MPSGraphTensor *product =
                    [graph matrixMultiplicationWithPrimaryTensor:tin
                                                 secondaryTensor:rhs name:nil];
                MPSGraphTensor *cast = [graph castTensor:product toType:type name:nil];

                id<MTLBuffer> ib = type == MPSDataTypeFloat32 ? in_f32 :
                    type == MPSDataTypeFloat16 ? in_f16 : in_bf16;
                id<MTLBuffer> wb;
                if (type == MPSDataTypeFloat32) wb = w_nk_f32;
                else if (type == MPSDataTypeFloat16)
                    wb = transposed ? w_nk_f16 : w_kn_f16;
                else wb = transposed ? w_nk_bf16 : w_kn_bf16;
                if (type == MPSDataTypeFloat32 && !transposed) continue;
                id<MTLBuffer> ob = type == MPSDataTypeFloat32 ? out32 : out16;
                size_t esize = type == MPSDataTypeFloat32 ? 4 : 2;

                MPSGraphTensorData *din = [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:ib shape:inShape dataType:type];
                MPSGraphTensorData *dw = [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:wb shape:wShape dataType:type];
                MPSGraphTensorData *dout = [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:ob shape:outShape dataType:type];
                double per;
                {
                    int warmups = 2, iterations = 5;
                    double wall = 0.0;
                    per = -1.0;
                    for (int i = 0; i < warmups + iterations; i++) {
                        @autoreleasepool {
                            MPSCommandBuffer *mps = [MPSCommandBuffer
                                commandBufferFromCommandQueue:queue];
                            double start = now_seconds();
                            [graph encodeToCommandBuffer:mps
                                feeds:@{tin: din, tw: dw}
                                targetOperations:nil
                                resultsDictionary:@{cast: dout}
                                executionDescriptor:nil];
                            [mps commitAndContinue];
                            id<MTLCommandBuffer> tail = mps.rootCommandBuffer;
                            [tail commit];
                            [tail waitUntilCompleted];
                            if (i >= warmups) wall += now_seconds() - start;
                        }
                    }
                    per = wall / iterations;
                }
                /* numeric spot check on out[3][5] */
                {
                    id<MTLCommandBuffer> command = [queue commandBuffer];
                    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
                    [blit copyFromBuffer:ob sourceOffset:((size_t)3 * N + 5) * esize
                        toBuffer:check destinationOffset:0 size:esize];
                    [blit endEncoding];
                    [command commit];
                    [command waitUntilCompleted];
                }
                float got = 0;
                if (type == MPSDataTypeFloat32) memcpy(&got, check.contents, 4);
                else if (type == MPSDataTypeFloat16) {
                    __fp16 raw;
                    memcpy(&raw, check.contents, 2);
                    got = (float)raw;
                } else {
                    uint16_t raw;
                    memcpy(&raw, check.contents, 2);
                    got = bf16_to_f32(raw);
                }
                double rel = fabs(got - reference) /
                    (fabs(reference) > 1e-6 ? fabs(reference) : 1.0);
                printf("  %-13s %8.1f ms  %6.2f TFLOP/s  spot rel_err %.4f%s\n",
                       gvs[g].label, per * 1e3, flops / per / 1e12, rel,
                       rel > 0.05 ? "  <-- WRONG?" : "");
                if (s == 0) names[variant_count + vi] = gvs[g].label;
                totals[vi++] += per;
            }

            /* --- production-shaped FP16: BF16 activations scaled 2^-3 and
             *     cast in-graph, weight buffer FP16 [N,K] pre-scaled 2^-3,
             *     output cast back to BF16 and rescaled 2^6. This is the
             *     exact overflow-guarded graph the h3 FP16 path runs. --- */
            {
                MPSGraph *graph = [[MPSGraph alloc] init];
                NSArray *inShape = @[@1, @(rows), @(K)];
                NSArray *wShape = @[@1, @(N), @(K)];
                NSArray *outShape = @[@1, @(rows), @(N)];
                MPSGraphTensor *tin = [graph placeholderWithShape:inShape
                    dataType:MPSDataTypeBFloat16 name:nil];
                MPSGraphTensor *tw = [graph placeholderWithShape:wShape
                    dataType:MPSDataTypeFloat16 name:nil];
                MPSGraphTensor *scaled_in = [graph
                    multiplicationWithPrimaryTensor:tin
                    secondaryTensor:[graph constantWithScalar:0.125
                        dataType:MPSDataTypeBFloat16] name:nil];
                MPSGraphTensor *cast_in = [graph castTensor:scaled_in
                    toType:MPSDataTypeFloat16 name:nil];
                MPSGraphTensor *rhs = [graph transposeTensor:tw dimension:1
                    withDimension:2 name:nil];
                MPSGraphTensor *product =
                    [graph matrixMultiplicationWithPrimaryTensor:cast_in
                                                 secondaryTensor:rhs name:nil];
                MPSGraphTensor *wide = [graph castTensor:product
                    toType:MPSDataTypeBFloat16 name:nil];
                MPSGraphTensor *cast = [graph
                    multiplicationWithPrimaryTensor:wide
                    secondaryTensor:[graph constantWithScalar:64.0
                        dataType:MPSDataTypeBFloat16] name:nil];
                MPSGraphTensorData *din = [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:in_bf16 shape:inShape
                    dataType:MPSDataTypeBFloat16];
                MPSGraphTensorData *dw = [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:w_nk_f16s shape:wShape
                    dataType:MPSDataTypeFloat16];
                MPSGraphTensorData *dout = [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:out16 shape:outShape
                    dataType:MPSDataTypeBFloat16];
                double per;
                {
                    int warmups = 2, iterations = 5;
                    double wall = 0.0;
                    for (int i = 0; i < warmups + iterations; i++) {
                        @autoreleasepool {
                            MPSCommandBuffer *mps = [MPSCommandBuffer
                                commandBufferFromCommandQueue:queue];
                            double start = now_seconds();
                            [graph encodeToCommandBuffer:mps
                                feeds:@{tin: din, tw: dw}
                                targetOperations:nil
                                resultsDictionary:@{cast: dout}
                                executionDescriptor:nil];
                            [mps commitAndContinue];
                            id<MTLCommandBuffer> tail = mps.rootCommandBuffer;
                            [tail commit];
                            [tail waitUntilCompleted];
                            if (i >= warmups) wall += now_seconds() - start;
                        }
                    }
                    per = wall / iterations;
                }
                {
                    id<MTLCommandBuffer> command = [queue commandBuffer];
                    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
                    [blit copyFromBuffer:out16 sourceOffset:((size_t)3 * N + 5) * 2
                        toBuffer:check destinationOffset:0 size:2];
                    [blit endEncoding];
                    [command commit];
                    [command waitUntilCompleted];
                }
                uint16_t raw;
                memcpy(&raw, check.contents, 2);
                float got = bf16_to_f32(raw);
                double rel = fabs(got - reference) /
                    (fabs(reference) > 1e-6 ? fabs(reference) : 1.0);
                printf("  %-13s %8.1f ms  %6.2f TFLOP/s  spot rel_err %.4f%s\n",
                       "h3-f16-path", per * 1e3, flops / per / 1e12, rel,
                       rel > 0.05 ? "  <-- WRONG?" : "");
                if (s == 0) names[variant_count + vi] = "h3-f16-path";
                totals[vi++] += per;

                /* Full-surface verification: a lone spot check cannot see a
                 * regionally-broken kernel (the SDPA int32 bug taught us).
                 * Pull the whole output back and check 20k random cells
                 * against the CPU. Requires the freed host arrays, so redo
                 * the generators with the same seed. */
                {
                    id<MTLBuffer> full = [device newBufferWithLength:o_elems * 2
                        options:MTLResourceStorageModeShared];
                    id<MTLCommandBuffer> command = [queue commandBuffer];
                    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
                    [blit copyFromBuffer:out16 sourceOffset:0 toBuffer:full
                        destinationOffset:0 size:o_elems * 2];
                    [blit endEncoding];
                    [command commit];
                    [command waitUntilCompleted];
                    float *a_check = malloc(a_elems * sizeof(float));
                    float *w_check = malloc(w_elems * sizeof(float));
                    srand(42 + (int)s);
                    for (size_t i = 0; i < a_elems; i++) {
                        float value = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
                        uint16_t b = f32_to_bf16(value);
                        a_check[i] = bf16_to_f32(b);
                    }
                    for (size_t i = 0; i < w_elems; i++) {
                        float value = ((float)rand() / RAND_MAX - 0.5f) * 0.06f;
                        uint16_t b = f32_to_bf16(value);
                        w_check[i] = bf16_to_f32(b);
                    }
                    const uint16_t *result = full.contents;
                    srand(1234);
                    size_t bad = 0, checked = 20000;
                    double worst = 0.0;
                    size_t worst_row = 0, worst_col = 0;
                    for (size_t probe_i = 0; probe_i < checked; probe_i++) {
                        size_t row = (size_t)rand() % rows;
                        size_t col = (size_t)rand() % N;
                        float expect = cpu_dot(a_check, w_check, K,
                                               (uint32_t)row, (uint32_t)col, K);
                        float have = bf16_to_f32(result[row * N + col]);
                        double err = fabs(have - expect) /
                            (fabs(expect) > 1e-3 ? fabs(expect) : 1e-3);
                        if (err > worst) {
                            worst = err;
                            worst_row = row;
                            worst_col = col;
                        }
                        if (err > 0.05) bad++;
                    }
                    printf("  %-13s full-check: %zu/%zu cells >5%% err, "
                           "worst %.4f at [%zu,%zu]%s\n", "",
                           bad, checked, worst, worst_row, worst_col,
                           bad ? "  <-- REGIONALLY BROKEN" : "");
                    free(a_check);
                    free(w_check);
                }
            }

            /* --- MPSMatrixMultiplication variants --- */
            struct { const char *label; int fp32; } mvs[] = {
                {"mps-f16", 0}, {"mps-f32", 1},
            };
            for (size_t m = 0; m < sizeof(mvs) / sizeof(mvs[0]); m++) {
                int fp32 = mvs[m].fp32;
                size_t esize = fp32 ? 4 : 2;
                MPSDataType mtype = fp32 ? MPSDataTypeFloat32 : MPSDataTypeFloat16;
                MPSMatrixDescriptor *da = [MPSMatrixDescriptor
                    matrixDescriptorWithRows:rows columns:K
                    rowBytes:(size_t)K * esize dataType:mtype];
                MPSMatrixDescriptor *dw = [MPSMatrixDescriptor
                    matrixDescriptorWithRows:N columns:K
                    rowBytes:(size_t)K * esize dataType:mtype];
                MPSMatrixDescriptor *dc = [MPSMatrixDescriptor
                    matrixDescriptorWithRows:rows columns:N
                    rowBytes:(size_t)N * esize dataType:mtype];
                MPSMatrix *ma = [[MPSMatrix alloc]
                    initWithBuffer:(fp32 ? in_f32 : in_f16) descriptor:da];
                MPSMatrix *mw = [[MPSMatrix alloc]
                    initWithBuffer:(fp32 ? w_nk_f32 : w_nk_f16) descriptor:dw];
                MPSMatrix *mc = [[MPSMatrix alloc]
                    initWithBuffer:(fp32 ? out32 : out16) descriptor:dc];
                MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc]
                    initWithDevice:device transposeLeft:NO transposeRight:YES
                    resultRows:rows resultColumns:N interiorColumns:K
                    alpha:1.0 beta:0.0];
                double per = time_variant(queue, ^(id<MTLCommandBuffer> command) {
                    [mm encodeToCommandBuffer:command leftMatrix:ma
                        rightMatrix:mw resultMatrix:mc];
                }, 2, 5);
                {
                    id<MTLCommandBuffer> command = [queue commandBuffer];
                    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
                    [blit copyFromBuffer:(fp32 ? out32 : out16)
                        sourceOffset:((size_t)3 * N + 5) * esize
                        toBuffer:check destinationOffset:0 size:esize];
                    [blit endEncoding];
                    [command commit];
                    [command waitUntilCompleted];
                }
                float got = 0;
                if (fp32) memcpy(&got, check.contents, 4);
                else {
                    __fp16 raw;
                    memcpy(&raw, check.contents, 2);
                    got = (float)raw;
                }
                double rel = fabs(got - reference) /
                    (fabs(reference) > 1e-6 ? fabs(reference) : 1.0);
                printf("  %-13s %8.1f ms  %6.2f TFLOP/s  spot rel_err %.4f%s\n",
                       mvs[m].label, per * 1e3, flops / per / 1e12, rel,
                       per < 0 ? "  <-- FAILED" : rel > 0.05 ? "  <-- WRONG?" : "");
                if (s == 0) names[variant_count + vi] = mvs[m].label;
                totals[vi++] += per;
            }

            /* --- portable 16x16 tile shader --- */
            if (tile) {
                struct { uint32_t rows, k, n, has_bias; } args = {rows, K, N, 0};
                double per = time_variant(queue, ^(id<MTLCommandBuffer> command) {
                    id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
                    [enc setComputePipelineState:tile];
                    [enc setBuffer:in_bf16 offset:0 atIndex:0];
                    [enc setBuffer:w_nk_bf16 offset:0 atIndex:1];
                    [enc setBuffer:in_bf16 offset:0 atIndex:2];
                    [enc setBuffer:out16 offset:0 atIndex:3];
                    [enc setBytes:&args length:sizeof(args) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake((N + 15) / 16,
                                                          (rows + 15) / 16, 1)
                        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
                    [enc endEncoding];
                }, 1, 3);
                {
                    id<MTLCommandBuffer> command = [queue commandBuffer];
                    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
                    [blit copyFromBuffer:out16 sourceOffset:((size_t)3 * N + 5) * 2
                        toBuffer:check destinationOffset:0 size:2];
                    [blit endEncoding];
                    [command commit];
                    [command waitUntilCompleted];
                }
                uint16_t raw;
                memcpy(&raw, check.contents, 2);
                float got = bf16_to_f32(raw);
                double rel = fabs(got - reference) /
                    (fabs(reference) > 1e-6 ? fabs(reference) : 1.0);
                printf("  %-13s %8.1f ms  %6.2f TFLOP/s  spot rel_err %.4f%s\n",
                       "tile-bf16", per * 1e3, flops / per / 1e12, rel,
                       rel > 0.05 ? "  <-- WRONG?" : "");
                if (s == 0) names[variant_count + vi] = "tile-bf16";
                totals[vi++] += per;
            }
            if (s == 0) variant_count = vi;
        }

        printf("\n== per-block totals (4 GEMMs, rows=%u) ==\n", rows);
        double block_flops = 2.0 * rows *
            (5376.0 * 21504 + 7168.0 * 5376 + 5376.0 * 28672 + 14336.0 * 5376);
        for (int v = 0; v < variant_count; v++) {
            if (!names[v]) continue;
            printf("  %-13s %8.1f ms/block  -> %6.1f s per 4-step generation "
                   "(200 blocks)  %5.2f TFLOP/s\n",
                   names[v], totals[v] * 1e3, totals[v] * 200.0,
                   block_flops / totals[v] / 1e12);
        }
    }
    return 0;
}
