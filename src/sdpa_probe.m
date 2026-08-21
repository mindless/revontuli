// sdpa_probe.m -- does MPSGraph's native SDPA return correct results on this
// device at large sequence lengths?
//
// Context (logs/51-INVESTIGATION-STATE.md): h3 generations corrupt above some
// total-token threshold between 3553 and 6688 video tokens, resolution- and
// frame-count-independent. The attention scores matrix [1, 56, seq, seq] is the
// only intermediate in the whole model that scales with seq^2, it lives entirely
// inside MPSGraph where no h3 counter can see it, and its size crosses this
// device's maxBufferLength (3.5 GiB) and/or 2^31 inside exactly that window:
//
//   FP32 scores vs 3.5 GiB maxbuf  -> breaks at seq 4096 exactly
//   BF16 scores, int32 byte offset -> breaks at seq ~4379
//   BF16 scores vs 3.5 GiB maxbuf  -> breaks at seq ~5793
//   int32 element count            -> breaks at seq ~6193
//
// This probe rebuilds h3's exact SDPA graph (h3_gpu.m:2009 h3_gpu_sdpa_graph:
// native scaledDotProductAttention, BF16, batch 1, heads 56, head_dim 128,
// non-causal, encodeToCommandBuffer on an MPSCommandBuffer, private storage)
// and sweeps seq, checking a spot CPU reference. Where it first fails -- if it
// fails -- identifies the mechanism by the numbers above.
//
// Build:
//   xcrun clang -fobjc-arc -O2 -framework Foundation -framework Metal \
//     -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
//     src/sdpa_probe.m -o bin/sdpa_probe
//
// Usage:
//   ./bin/sdpa_probe [--head-major] [--shared] N [N ...]
//
// The first N should be ~1463 (the verified 22-frame shape) as a control: if
// the probe fails there, the probe is wrong, not the model.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <math.h>

enum { HEADS = 56, HEAD_DIM = 128 };

static uint16_t f32_to_bf16(float f) {
    uint32_t bits; memcpy(&bits, &f, 4);
    bits += 0x7FFFu + ((bits >> 16) & 1u);   /* round to nearest even */
    return (uint16_t)(bits >> 16);
}
static float bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16; float f; memcpy(&f, &bits, 4); return f;
}

/* Deterministic per-element value, independent of memory layout, so the CPU
 * reference and both GPU layouts read identical logical data. splitmix64. */
static float element_value(uint32_t tensor, uint32_t head, uint32_t s, uint32_t d) {
    uint64_t x = ((uint64_t)tensor << 48) ^ ((uint64_t)head << 36) ^
                 ((uint64_t)s << 12) ^ (uint64_t)d;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    float raw = (float)((x >> 40) & 0xFFFFFFu) / 8388608.0f - 1.0f; /* [-1,1) */
    return bf16_to_f32(f32_to_bf16(raw));  /* what BF16 storage will hold */
}

/* --spike mode: one planted high-logit key per probed head. K[h, spike_pos[h]]
 * is set parallel to Q[h,0] with a large amplitude, so query 0 of head h MUST
 * retrieve ~V[h, spike_pos[h]] (magnitude O(1)). If the GPU drops or misweights
 * a region of the key sequence, exactly the spikes planted in that region fail
 * with O(1) error -- the per-head pattern maps the broken region. */
enum { SPIKES = HEADS };  /* one spike per head -> full head map */
static uint32_t spike_pos[SPIKES];
static int spike_enabled = 0;

static float element_value_spiked(uint32_t tensor, uint32_t head, uint32_t s,
                                  uint32_t d) {
    if (spike_enabled && tensor == 1 && head < SPIKES && s == spike_pos[head])
        return bf16_to_f32(f32_to_bf16(8.0f * element_value(0, head, 0, d)));
    return element_value(tensor, head, s, d);
}

static size_t element_index(int headMajor, uint32_t seq,
                            uint32_t head, uint32_t s, uint32_t d) {
    if (headMajor) return (((size_t)head * seq) + s) * HEAD_DIM + d;
    return (((size_t)s * HEADS) + head) * HEAD_DIM + d;
}

int main(int argc, char **argv) {
    int headMajor = 0, shared = 0, split = 0;
    uint32_t seqs[64]; int nseq = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--head-major")) headMajor = 1;
        else if (!strcmp(argv[i], "--split")) split = 1;
        else if (!strcmp(argv[i], "--spike")) spike_enabled = 1;
        else if (!strcmp(argv[i], "--shared")) shared = 1;
        else if (nseq < 64) seqs[nseq++] = (uint32_t)strtoul(argv[i], NULL, 10);
    }
    if (!nseq) {
        fprintf(stderr, "usage: sdpa_probe [--head-major] [--shared] N [N ...]\n");
        return 64;
    }

    @autoreleasepool {
        id<MTLDevice> device = nil;
        NSArray<id<MTLDevice>> *all = MTLCopyAllDevices();
        const char *want = getenv("H3_GPU_NAME");
        NSString *needle = want && *want ? [NSString stringWithUTF8String:want]
                                         : @"6900";
        for (id<MTLDevice> candidate in all)
            if ([candidate.name containsString:needle]) { device = candidate; break; }
        if (!device) { fprintf(stderr, "no Metal device matching '%s'\n",
                               needle.UTF8String); return 70; }
        printf("device: %s  maxBufferLength=%.2f GiB  layout=%s  storage=%s\n",
               device.name.UTF8String,
               (double)device.maxBufferLength / (1024.0 * 1024.0 * 1024.0),
               headMajor ? "head-major" : "row-major",
               shared ? "shared" : "private");
        if (split) printf("head-split fix ENABLED\n");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        float scale = 1.0f / sqrtf((float)HEAD_DIM);

        for (int run = 0; run < nseq; run++) {
            uint32_t seq = seqs[run];
            size_t count = (size_t)seq * HEADS * HEAD_DIM;
            size_t bytes = count * sizeof(uint16_t);
            double scoresGiBbf16 = (double)HEADS * seq * seq * 2.0 /
                                   (1024.0 * 1024.0 * 1024.0);

            if (spike_enabled)
                /* Unique position per head, spread across the sequence. */
                for (int i = 0; i < SPIKES; i++)
                    spike_pos[i] = (uint32_t)(((uint64_t)(2 * i + 1) * seq) /
                                              (2 * SPIKES));
            @autoreleasepool {
                /* Host-side fill. */
                uint16_t *host[3];
                for (int t = 0; t < 3; t++) {
                    host[t] = malloc(bytes);
                    if (!host[t]) { fprintf(stderr, "oom\n"); return 71; }
                    for (uint32_t h = 0; h < HEADS; h++)
                        for (uint32_t s = 0; s < seq; s++)
                            for (uint32_t d = 0; d < HEAD_DIM; d++)
                                host[t][element_index(headMajor, seq, h, s, d)] =
                                    f32_to_bf16(element_value_spiked(
                                        (uint32_t)t, h, s, d));
                }

                /* Device tensors: private + blit upload to mirror production
                 * (activations live in VRAM), or shared with --shared. */
                MTLResourceOptions opt = shared ? MTLResourceStorageModeShared
                                                : MTLResourceStorageModePrivate;
                id<MTLBuffer> q, k, v, out;
                id<MTLBuffer> readback = nil;
                if (shared) {
                    q = [device newBufferWithBytes:host[0] length:bytes options:opt];
                    k = [device newBufferWithBytes:host[1] length:bytes options:opt];
                    v = [device newBufferWithBytes:host[2] length:bytes options:opt];
                    out = [device newBufferWithLength:bytes options:opt];
                } else {
                    q = [device newBufferWithLength:bytes options:opt];
                    k = [device newBufferWithLength:bytes options:opt];
                    v = [device newBufferWithLength:bytes options:opt];
                    out = [device newBufferWithLength:bytes options:opt];
                    readback = [device newBufferWithLength:bytes
                        options:MTLResourceStorageModeShared];
                    id<MTLBuffer> stage[3];
                    id<MTLCommandBuffer> upload = [queue commandBuffer];
                    id<MTLBlitCommandEncoder> blit = [upload blitCommandEncoder];
                    id<MTLBuffer> destinations[3] = {q, k, v};
                    for (int t = 0; t < 3; t++) {
                        stage[t] = [device newBufferWithBytes:host[t] length:bytes
                            options:MTLResourceStorageModeShared];
                        [blit copyFromBuffer:stage[t] sourceOffset:0
                                    toBuffer:destinations[t] destinationOffset:0
                                        size:bytes];
                    }
                    [blit endEncoding];
                    [upload commit];
                    [upload waitUntilCompleted];
                }

                /* The graph, exactly as h3_gpu_sdpa_graph builds it. */
                MPSGraph *graph = [[MPSGraph alloc] init];
                NSArray<NSNumber *> *rowMajorShape =
                    @[@1, @(seq), @(HEADS), @(HEAD_DIM)];
                NSArray<NSNumber *> *headMajorShape =
                    @[@1, @(HEADS), @(seq), @(HEAD_DIM)];
                NSArray<NSNumber *> *inputShape = headMajor ? headMajorShape
                                                            : rowMajorShape;
                MPSGraphTensor *qp = [graph placeholderWithShape:inputShape
                    dataType:MPSDataTypeBFloat16 name:nil];
                MPSGraphTensor *kp = [graph placeholderWithShape:inputShape
                    dataType:MPSDataTypeBFloat16 name:nil];
                MPSGraphTensor *vp = [graph placeholderWithShape:inputShape
                    dataType:MPSDataTypeBFloat16 name:nil];
                MPSGraphTensor *qt = headMajor ? qp :
                    [graph transposeTensor:qp dimension:1 withDimension:2 name:nil];
                MPSGraphTensor *kt = headMajor ? kp :
                    [graph transposeTensor:kp dimension:1 withDimension:2 name:nil];
                MPSGraphTensor *vt = headMajor ? vp :
                    [graph transposeTensor:vp dimension:1 withDimension:2 name:nil];
                /* THE FIX UNDER TEST: when heads*seq*seq crosses 2^31 the
                 * one-op SDPA corrupts (int32 overflow in MPS's internal scores
                 * addressing, exact boundary measured at seq 6193 for 56 heads).
                 * Split into head chunks so each chunk's scores tensor stays
                 * comfortably below 2^31 elements. */
                MPSGraphTensor *attention;
                size_t scores_elements = (size_t)HEADS * seq * seq;
                if (split && scores_elements > 1900000000UL) {
                    uint32_t chunk = HEADS;
                    while ((size_t)chunk * seq * seq > 1900000000UL)
                        chunk = (chunk + 1) / 2;
                    NSMutableArray<MPSGraphTensor *> *pieces =
                        [NSMutableArray array];
                    for (uint32_t at = 0; at < HEADS; at += chunk) {
                        uint32_t take = at + chunk <= HEADS ? chunk : HEADS - at;
                        MPSGraphTensor *qs = [graph sliceTensor:qt dimension:1
                            start:at length:take name:nil];
                        MPSGraphTensor *ks = [graph sliceTensor:kt dimension:1
                            start:at length:take name:nil];
                        MPSGraphTensor *vs = [graph sliceTensor:vt dimension:1
                            start:at length:take name:nil];
                        [pieces addObject:[graph
                            scaledDotProductAttentionWithQueryTensor:qs
                            keyTensor:ks valueTensor:vs scale:scale name:nil]];
                    }
                    attention = pieces.count == 1 ? pieces[0] :
                        [graph concatTensors:pieces dimension:1 name:nil];
                } else {
                    attention = [graph
                        scaledDotProductAttentionWithQueryTensor:qt keyTensor:kt
                        valueTensor:vt scale:scale name:nil];
                }
                MPSGraphTensor *result = headMajor ? attention :
                    [graph transposeTensor:attention dimension:1 withDimension:2
                                      name:nil];

                MPSGraphTensorData *(^wrap)(id<MTLBuffer>) =
                    ^MPSGraphTensorData *(id<MTLBuffer> buffer) {
                        return [[MPSGraphTensorData alloc]
                            initWithMTLBuffer:buffer shape:inputShape
                                     dataType:MPSDataTypeBFloat16];
                    };

                /* Encode through MPSCommandBuffer like production, so MPSGraph
                 * is free to commitAndContinue underneath us. */
                id<MTLCommandBuffer> root = [queue commandBuffer];
                MPSCommandBuffer *command =
                    [MPSCommandBuffer commandBufferWithCommandBuffer:root];
                double t0 = (double)clock() / CLOCKS_PER_SEC;
                @try {
                    [graph encodeToCommandBuffer:command
                        feeds:@{qp: wrap(q), kp: wrap(k), vp: wrap(v)}
                        targetOperations:nil
                        resultsDictionary:@{result: wrap(out)}
                        executionDescriptor:nil];
                } @catch (NSException *exception) {
                    printf("seq=%5u  ENCODE THREW: %s\n", seq,
                           exception.reason.UTF8String);
                    for (int t = 0; t < 3; t++) free(host[t]);
                    continue;
                }
                double encode_s = (double)clock() / CLOCKS_PER_SEC - t0;
                root = command.rootCommandBuffer;
                if (readback) {
                    id<MTLBlitCommandEncoder> blit = [root blitCommandEncoder];
                    [blit copyFromBuffer:out sourceOffset:0 toBuffer:readback
                        destinationOffset:0 size:bytes];
                    [blit endEncoding];
                }
                NSDate *walltime = [NSDate date];
                [root commit];
                [root waitUntilCompleted];
                double wait_s = -walltime.timeIntervalSinceNow;
                if (root.status == MTLCommandBufferStatusError) {
                    printf("seq=%5u  COMMAND BUFFER ERROR: %s\n", seq,
                           root.error.localizedDescription.UTF8String);
                    for (int t = 0; t < 3; t++) free(host[t]);
                    continue;
                }

                const uint16_t *o = (readback ? readback : out).contents;

                /* Invariants over the full output: no NaN/Inf, and every value
                 * inside the convex hull of V (|out| <= max|V| = 1). */
                size_t nans = 0, hull = 0;
                float peak = 0.0f;
                for (size_t i = 0; i < count; i++) {
                    float f = bf16_to_f32(o[i]);
                    if (isnan(f) || isinf(f)) { nans++; continue; }
                    float a = fabsf(f);
                    if (a > peak) peak = a;
                    if (a > 1.02f) hull++;
                }

                /* Spot CPU reference: 5 query positions x 3 heads, softmax in
                 * double over the full key length. */
                uint32_t spot_s[5] = {0, 1, seq / 2, seq - 2, seq - 1};
                uint32_t spot_h[3] = {0, HEADS / 2, HEADS - 1};
                uint32_t spike_q[3] = {0, seq / 2, seq - 1};
                int nspots = spike_enabled ? HEADS * 3 : 15;
                int broken = 0;
                double max_err = 0.0;
                double *logits = malloc((size_t)seq * sizeof(*logits));
                for (int spot = 0; spot < nspots; spot++) {
                    uint32_t s, h;
                    if (spike_enabled) { h = (uint32_t)(spot / 3);
                                         s = spike_q[spot % 3]; }
                    else { s = spot_s[spot / 3]; h = spot_h[spot % 3]; }
                    double spot_err = 0.0;
                    double lmax = -1e300;
                    for (uint32_t j = 0; j < seq; j++) {
                        double dot = 0.0;
                        for (uint32_t d = 0; d < HEAD_DIM; d++)
                            dot += (double)element_value_spiked(0, h, s, d) *
                                   (double)element_value_spiked(1, h, j, d);
                        logits[j] = dot * scale;
                        if (logits[j] > lmax) lmax = logits[j];
                    }
                    double denom = 0.0;
                    for (uint32_t j = 0; j < seq; j++) {
                        logits[j] = exp(logits[j] - lmax);
                        denom += logits[j];
                    }
                    for (uint32_t d = 0; d < HEAD_DIM; d++) {
                        double acc = 0.0;
                        for (uint32_t j = 0; j < seq; j++)
                            acc += logits[j] *
                                   (double)element_value_spiked(2, h, j, d);
                        double reference = acc / denom;
                        double got = (double)bf16_to_f32(
                            o[element_index(headMajor, seq, h, s, d)]);
                        double err = fabs(got - reference);
                        if (err > max_err) max_err = err;
                        if (err > spot_err) spot_err = err;
                    }
                    if (spike_enabled && spot_err > 0.05) {
                        printf("  BROKEN head=%2u q=%5u j*=%5u err=%.4f\n",
                               h, s, spike_pos[h], spot_err);
                        broken++;
                    }
                }
                free(logits);

                if (spike_enabled)
                    printf("  -> %d of %d (head,query) spots broken\n",
                           broken, nspots);
                printf("seq=%5u  scores(bf16)=%6.2f GiB  encode=%7.3fs  "
                       "gpu-wall=%7.3fs  nan=%zu  hull>1=%zu  peak=%.4f  "
                       "spot_max_err=%.6f  %s\n",
                       seq, scoresGiBbf16, encode_s, wait_s, nans, hull,
                       (double)peak, max_err,
                       (nans || hull || max_err > 0.05) ? "*** FAIL ***" : "ok");
                fflush(stdout);
                for (int t = 0; t < 3; t++) free(host[t]);
            }
        }
    }
    return 0;
}
