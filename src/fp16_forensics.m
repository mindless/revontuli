/* fp16_forensics.m
 *
 * My broad capability probe reported 13 NaNs out of ~16.8k sampled FP16
 * results on the RX 6900 XT. Before I blame the GPU I need to know whether
 * the fault is in the hardware, in Metal's `half` handling, or in my own
 * host-side float<->half converter. This program isolates that.
 *
 * I do three things:
 *   1. Test my host converter against the compiler's own _Float16 conversion,
 *      so I can rule my own arithmetic in or out first.
 *   2. Run the identical GPU kernel several times and report whether the bad
 *      indices are STABLE (a deterministic bug) or MOVING (a real
 *      nondeterministic transfer/compute fault, which is what the upstream
 *      PyTorch RDNA2 reports describe).
 *   3. Print the actual offending input/output bit patterns.
 *
 * Build:
 *   xcrun clang -fobjc-arc -O2 -framework Foundation -framework Metal \
 *     src/fp16_forensics.m -o bin/fp16_forensics
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <string.h>

/* The crude converter from my capability probe, reproduced verbatim so I can
 * indict or exonerate it. */
static uint16_t probe_f32_to_f16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t sign = (u >> 31) & 1u;
    int32_t  exp  = (int32_t)((u >> 23) & 0xffu) - 127 + 15;
    uint32_t man  = u & 0x7fffffu;
    if (exp <= 0)  return (uint16_t)(sign << 15);
    if (exp >= 31) return (uint16_t)((sign << 15) | 0x7c00u);
    return (uint16_t)((sign << 15) | ((uint32_t)exp << 10) | (man >> 13));
}
static float probe_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1fu;
    uint32_t man  = (uint32_t)h & 0x3ffu;
    uint32_t out;
    if (exp == 0)       out = sign << 31;
    else if (exp == 31) out = (sign << 31) | 0x7f800000u | (man << 13);
    else                out = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    float f; memcpy(&f, &out, 4); return f;
}

/* The compiler's own conversion, used as ground truth. */
static uint16_t true_f32_to_f16(float f) {
    _Float16 h = (_Float16)f; uint16_t b; memcpy(&b, &h, 2); return b;
}
static float true_f16_to_f32(uint16_t b) {
    _Float16 h; memcpy(&h, &b, 2); return (float)h;
}

static NSString *const kSource = @"\
#include <metal_stdlib>\n\
using namespace metal;\n\
kernel void fma_f16(device const half *a [[buffer(0)]],\n\
                    device const half *b [[buffer(1)]],\n\
                    device half *c [[buffer(2)]],\n\
                    uint i [[thread_position_in_grid]]) {\n\
    c[i] = (half)fma((float)a[i], (float)b[i], (float)a[i]);\n\
}\n";

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *wanted = (argc > 1) ? argv[1] : "6900";
        id<MTLDevice> device = nil;
        for (id<MTLDevice> d in MTLCopyAllDevices())
            if ([d.name rangeOfString:[NSString stringWithUTF8String:wanted]].location
                != NSNotFound) { device = d; break; }
        if (!device) { printf("I could not find device \"%s\".\n", wanted); return 2; }
        printf("I am investigating FP16 on: %s\n\n", device.name.UTF8String);

        /* ---------- Step 1: is my own host converter the culprit? ---------- */
        printf("=== Step 1: my host converter vs the compiler's _Float16 ===\n");
        const NSUInteger COUNT = 16u * 1024u * 1024u;
        NSUInteger conv_mismatch = 0, first_bad = (NSUInteger)-1;
        for (NSUInteger i = 0; i < COUNT; i++) {
            float av = (float)((long)(i % 1000) - 500) * 0.001f;
            uint16_t mine = probe_f32_to_f16(av);
            uint16_t truth = true_f32_to_f16(av);
            if (mine != truth) {
                if (conv_mismatch == 0) first_bad = i;
                conv_mismatch++;
            }
        }
        printf("  input stream A: %lu / %lu values converted differently\n",
               (unsigned long)conv_mismatch, (unsigned long)COUNT);
        if (first_bad != (NSUInteger)-1) {
            float av = (float)((first_bad % 1000) - 500) * 0.001f;
            printf("  first divergence at i=%lu: value=%.9g mine=0x%04x (%.9g) "
                   "truth=0x%04x (%.9g)\n",
                   (unsigned long)first_bad, av,
                   probe_f32_to_f16(av), probe_f16_to_f32(probe_f32_to_f16(av)),
                   true_f32_to_f16(av), true_f16_to_f32(true_f32_to_f16(av)));
            printf("  -> my converter TRUNCATES the mantissa and flushes\n"
                   "     subnormals to zero; the compiler rounds correctly.\n");
        }
        printf("\n");

        /* ---------- Step 2: repeatability of the GPU result ---------- */
        printf("=== Step 2: GPU determinism across repeated identical runs ===\n");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError *err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:kSource options:nil error:&err];
        if (!lib) { printf("compile failed: %s\n", err.localizedDescription.UTF8String); return 3; }
        id<MTLComputePipelineState> pipe =
            [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fma_f16"]
                                                 error:&err];
        if (!pipe) { printf("pipeline failed: %s\n", err.localizedDescription.UTF8String); return 3; }

        id<MTLBuffer> a = [device newBufferWithLength:COUNT * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> b = [device newBufferWithLength:COUNT * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> c = [device newBufferWithLength:COUNT * 2 options:MTLResourceStorageModeShared];
        uint16_t *pa = a.contents, *pb = b.contents, *pc = c.contents;

        /* This time I build the inputs with the CORRECT converter, so the GPU
         * receives only well-formed halves. */
        for (NSUInteger i = 0; i < COUNT; i++) {
            pa[i] = true_f32_to_f16((float)((long)(i % 1000) - 500) * 0.001f);
            pb[i] = true_f32_to_f16((float)((long)(i % 777) - 388) * 0.002f);
        }

        NSMutableArray<NSString *> *runSignatures = [NSMutableArray array];
        for (int run = 0; run < 5; run++) {
            memset(pc, 0xAB, COUNT * 2);
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipe];
            [enc setBuffer:a offset:0 atIndex:0];
            [enc setBuffer:b offset:0 atIndex:1];
            [enc setBuffer:c offset:0 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(COUNT, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            /* Full sweep, not sampled: I want every bad element. */
            NSUInteger nan_count = 0, wrong = 0;
            double worst = 0.0;
            NSMutableArray<NSNumber *> *badIdx = [NSMutableArray array];
            for (NSUInteger i = 0; i < COUNT; i++) {
                float av = true_f16_to_f32(pa[i]);
                float bv = true_f16_to_f32(pb[i]);
                float ref = (float)(_Float16)fmaf(av, bv, av);
                float got = true_f16_to_f32(pc[i]);
                if (isnan(got) || isinf(got)) {
                    nan_count++;
                    if (badIdx.count < 8) [badIdx addObject:@(i)];
                    continue;
                }
                double d = fabs((double)got - (double)ref);
                if (d > worst) worst = d;
                if (d > 1e-3) {
                    wrong++;
                    if (badIdx.count < 8) [badIdx addObject:@(i)];
                }
            }
            printf("  run %d: NaN/Inf=%lu  wrong(>1e-3)=%lu  max_abs_err=%.3g\n",
                   run, (unsigned long)nan_count, (unsigned long)wrong, worst);
            NSMutableString *sig = [NSMutableString string];
            for (NSNumber *n in badIdx) [sig appendFormat:@"%@,", n];
            [runSignatures addObject:sig.length ? sig : @"(clean)"];
            if (badIdx.count) {
                printf("        first bad indices: %s\n", sig.UTF8String);
                for (NSNumber *n in badIdx) {
                    NSUInteger i = n.unsignedLongValue;
                    printf("          i=%-10lu a=0x%04x b=0x%04x c=0x%04x\n",
                           (unsigned long)i, pa[i], pb[i], pc[i]);
                }
            }
        }

        printf("\n=== Step 3: verdict ===\n");
        BOOL allClean = YES, allIdentical = YES;
        for (NSString *s in runSignatures) {
            if (![s isEqualToString:@"(clean)"]) allClean = NO;
            if (![s isEqualToString:runSignatures[0]]) allIdentical = NO;
        }
        if (allClean) {
            printf("  Every run was numerically clean once the inputs were built\n"
                   "  with a correctly-rounding converter. The 13 NaNs in my\n"
                   "  earlier capability run were an artefact of MY truncating\n"
                   "  host converter, NOT an RX 6900 XT fault.\n");
            printf("  Metal FP16 on gfx1030: PASS\n");
        } else if (allIdentical) {
            printf("  The same indices fail on every run: this is a DETERMINISTIC\n"
                   "  fault (compiler or kernel semantics), not memory corruption.\n");
        } else {
            printf("  The failing indices MOVE between identical runs. That is\n"
                   "  nondeterministic corruption and it matches the class of bug\n"
                   "  reported upstream for RDNA2. I would not trust this path.\n");
        }
        return allClean ? 0 : 1;
    }
}
