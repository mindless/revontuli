/* linear_bw.m
 *
 * I run h3.c's OWN h3_linear_bf16 kernel, at MiniMax H3's real DiT dimensions,
 * with the weight matrix placed once in Shared (host) memory and once in
 * Private (VRAM) memory. This is the experiment that decides whether the GPU
 * timeout I hit in the denoise loop is a bandwidth problem I can fix by moving
 * weights into VRAM, or something deeper.
 *
 * H3 FL2VA DiT config: hidden_size 5376, heads 56 x 128 = 7168 inner,
 * ffn_hidden_size 14336. The QKV projection is therefore 5376 -> 21504.
 *
 * The kernel tiles 16x16 and reloads the weight tile for every 16-row block,
 * so the weight matrix is streamed (rows/16) times. That multiplier is what
 * turns a 231 MiB weight into many GiB of traffic.
 *
 * Build:
 *   xcrun clang -fobjc-arc -O2 -framework Foundation -framework Metal \
 *     src/linear_bw.m -o bin/linear_bw
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

typedef struct { uint32_t rows, input_dim, output_dim, has_bias; } linear_args;

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *shaders = (argc > 1) ? argv[1] : "src/h3.c/h3_shaders.metal";
        id<MTLDevice> dev = nil;
        for (id<MTLDevice> d in MTLCopyAllDevices())
            if ([d.name rangeOfString:@"6900"].location != NSNotFound) { dev = d; break; }
        if (!dev) { printf("no RX 6900 XT\n"); return 2; }
        id<MTLCommandQueue> q = [dev newCommandQueue];

        NSError *err = nil;
        NSString *src = [NSString stringWithContentsOfFile:
                            [NSString stringWithUTF8String:shaders]
                                                  encoding:NSUTF8StringEncoding
                                                     error:&err];
        MTLCompileOptions *opt = [[MTLCompileOptions alloc] init];
        opt.mathMode = MTLMathModeSafe;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opt error:&err];
        if (!lib) { printf("shader compile failed: %s\n",
                           err.localizedDescription.UTF8String); return 3; }
        id<MTLComputePipelineState> pipe =
            [dev newComputePipelineStateWithFunction:
                [lib newFunctionWithName:@"h3_linear_bf16"] error:&err];
        if (!pipe) { printf("pipeline failed\n"); return 3; }

        const uint32_t HIDDEN = 5376;
        const uint32_t INNER3 = 7168 * 3;          /* QKV projection width */
        const uint32_t FFN2   = 14336 * 2;         /* SwiGLU FC1 width     */

        struct { const char *label; uint32_t in_dim, out_dim; } shapes[] = {
            { "QKV  5376->21504", HIDDEN, INNER3 },
            { "FC1  5376->28672", HIDDEN, FFN2   },
        };
        uint32_t rowCounts[] = { 256, 1024 };

        printf("I am timing h3.c's own h3_linear_bf16 on %s\n", dev.name.UTF8String);
        printf("(macOS aborts a command buffer that makes no progress for ~5 s)\n\n");
        printf("  %-18s %6s %8s %10s %10s %9s\n",
               "shape", "rows", "weightMiB", "shared s", "private s", "speedup");
        printf("  %s\n", "-------------------------------------------------------------------------");

        for (unsigned s = 0; s < 2; s++) {
            for (unsigned r = 0; r < 2; r++) {
                uint32_t rows = rowCounts[r];
                uint32_t in_dim = shapes[s].in_dim, out_dim = shapes[s].out_dim;
                size_t wElems = (size_t)in_dim * out_dim;
                size_t wBytes = wElems * 2;

                id<MTLBuffer> input = [dev newBufferWithLength:(size_t)rows * in_dim * 2
                                                       options:MTLResourceStorageModeShared];
                id<MTLBuffer> output = [dev newBufferWithLength:(size_t)rows * out_dim * 2
                                                        options:MTLResourceStorageModeShared];
                /* A Shared weight, exactly what h3.c allocates today. */
                id<MTLBuffer> wShared = [dev newBufferWithLength:wBytes
                                                         options:MTLResourceStorageModeShared];
                if (!input || !output || !wShared) { printf("  alloc failed\n"); continue; }
                memset(wShared.contents, 0x3c, wBytes);   /* harmless BF16 pattern */
                memset(input.contents, 0x3c, (size_t)rows * in_dim * 2);

                /* The same weight staged into real VRAM. */
                id<MTLBuffer> wPrivate = [dev newBufferWithLength:wBytes
                                                          options:MTLResourceStorageModePrivate];
                if (wPrivate) {
                    id<MTLCommandBuffer> bcb = [q commandBuffer];
                    id<MTLBlitCommandEncoder> blit = [bcb blitCommandEncoder];
                    [blit copyFromBuffer:wShared sourceOffset:0
                                toBuffer:wPrivate destinationOffset:0 size:wBytes];
                    [blit endEncoding];
                    [bcb commit];
                    [bcb waitUntilCompleted];
                }

                linear_args args = { rows, in_dim, out_dim, 0u };
                double times[2] = {0, 0};
                for (int mode = 0; mode < 2; mode++) {
                    id<MTLBuffer> w = mode ? wPrivate : wShared;
                    if (!w) { times[mode] = -1; continue; }
                    NSDate *t0 = [NSDate date];
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pipe];
                    [enc setBuffer:input offset:0 atIndex:0];
                    [enc setBuffer:w offset:0 atIndex:1];
                    [enc setBuffer:input offset:0 atIndex:2];   /* bias unused */
                    [enc setBuffer:output offset:0 atIndex:3];
                    [enc setBytes:&args length:sizeof(args) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake((out_dim + 15) / 16,
                                                          (rows + 15) / 16, 1)
                        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
                    [enc endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                    times[mode] = -[t0 timeIntervalSinceNow];
                    if (cb.status == MTLCommandBufferStatusError) {
                        printf("      %s weight: COMMAND BUFFER ERROR: %s\n",
                               mode ? "private" : "shared",
                               cb.error.localizedDescription.UTF8String);
                        times[mode] = -1;
                    }
                }
                printf("  %-18s %6u %8.1f %10.3f %10.3f %8.0fx%s\n",
                       shapes[s].label, rows, (double)wBytes / 1048576.0,
                       times[0], times[1],
                       (times[0] > 0 && times[1] > 0) ? times[0] / times[1] : 0.0,
                       (times[0] > 5.0) ? "   <-- shared exceeds watchdog" : "");
            }
        }
        printf("\n  Effective weight traffic = weightMiB * ceil(rows/16), because the\n");
        printf("  16x16 tile kernel rereads the weight for every 16-row block.\n");
        return 0;
    }
}
