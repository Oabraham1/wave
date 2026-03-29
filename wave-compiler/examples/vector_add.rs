#[kernel]
fn vector_add(a: &[f32], b: &[f32], out: &mut [f32], n: u32) {
    let gid = thread_id();
    if gid < n {
        out[gid] = a[gid] + b[gid];
    }
}
