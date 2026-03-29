function vector_add(a: f32[], b: f32[], out: f32[], n: u32) {
    const gid = thread_id();
    if (gid < n) {
        out[gid] = a[gid] + b[gid];
    }
}
