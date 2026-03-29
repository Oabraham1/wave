__kernel void vector_add(float* a, float* b, float* out, uint32_t n) {
    uint32_t gid = thread_id();
    if (gid < n) {
        out[gid] = a[gid] + b[gid];
    }
}
