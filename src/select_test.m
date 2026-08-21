/* I link against the patched libh3.a and call the real selector, so I am
 * testing the exact code that ships in h3, not a copy of it. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "h3_device_select.h"
#include <stdio.h>
int main(void) { @autoreleasepool {
    id<MTLDevice> d = h3_select_metal_device();
    if (!d) { printf("RESULT: nil (refused to fall back)\n"); return 1; }
    printf("RESULT: %s\n", d.name.UTF8String);
    return 0;
}}
