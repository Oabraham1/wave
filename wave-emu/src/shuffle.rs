// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Wave shuffle and collective operations. Implements cross-lane communication, shuffle,
//! broadcast, vote operations, ballot, any, all, and reductions, prefix sum, reduce add/min/max.
//! Operates only on active threads.

use crate::wave::Wave;

pub fn wave_shuffle(wave: &mut Wave, rd: u8, rs1: u8, rs_lane: u8) {
    let wave_width = wave.wave_width;
    let active_mask = wave.active_mask;

    let mut results = [0u32; 64];

    for lane in 0..wave_width {
        if (active_mask & (1u64 << lane)) != 0 {
            let src_lane = wave.threads[lane as usize].read_register(rs_lane);
            if src_lane < wave_width {
                results[lane as usize] = wave.threads[src_lane as usize].read_register(rs1);
            }
        }
    }

    for lane in 0..wave_width {
        if (active_mask & (1u64 << lane)) != 0 {
            wave.threads[lane as usize].write_register(rd, results[lane as usize]);
        }
    }
}

pub fn wave_shuffle_up(wave: &mut Wave, rd: u8, rs1: u8, rs_delta: u8) {
    let wave_width = wave.wave_width;
    let active_mask = wave.active_mask;

    let mut results = [0u32; 64];

    for lane in 0..wave_width {
        if (active_mask & (1u64 << lane)) != 0 {
            let delta = wave.threads[lane as usize].read_register(rs_delta);
            if lane >= delta {
                let src_lane = lane - delta;
                results[lane as usize] = wave.threads[src_lane as usize].read_register(rs1);
            }
        }
    }

    for lane in 0..wave_width {
        if (active_mask & (1u64 << lane)) != 0 {
            wave.threads[lane as usize].write_register(rd, results[lane as usize]);
        }
    }
}

pub fn wave_shuffle_down(wave: &mut Wave, rd: u8, rs1: u8, rs_delta: u8) {
    let wave_width = wave.wave_width;
    let active_mask = wave.active_mask;

    let mut results = [0u32; 64];

    for lane in 0..wave_width {
        if (active_mask & (1u64 << lane)) != 0 {
            let delta = wave.threads[lane as usize].read_register(rs_delta);
            if let Some(src_lane) = lane.checked_add(delta).filter(|&src| src < wave_width) {
                results[lane as usize] = wave.threads[src_lane as usize].read_register(rs1);
            }
        }
    }

    for lane in 0..wave_width {
        if (active_mask & (1u64 << lane)) != 0 {
            wave.threads[lane as usize].write_register(rd, results[lane as usize]);
        }
    }
}

pub fn wave_shuffle_xor(wave: &mut Wave, rd: u8, rs1: u8, rs_mask: u8) {
    let wave_width = wave.wave_width;
    let active_mask = wave.active_mask;

    let mut results = [0u32; 64];

    for lane in 0..wave_width {
        if (active_mask & (1u64 << lane)) != 0 {
            let mask = wave.threads[lane as usize].read_register(rs_mask);
            let src_lane = lane ^ mask;
            if src_lane < wave_width {
                results[lane as usize] = wave.threads[src_lane as usize].read_register(rs1);
            }
        }
    }

    for lane in 0..wave_width {
        if (active_mask & (1u64 << lane)) != 0 {
            wave.threads[lane as usize].write_register(rd, results[lane as usize]);
        }
    }
}

pub fn wave_broadcast(wave: &mut Wave, rd: u8, rs1: u8, rs_lane: u8) {
    let wave_width = wave.wave_width;
    let active_mask = wave.active_mask;

    let mut src_lane = 0u32;
    for lane in 0..wave_width {
        if (active_mask & (1u64 << lane)) != 0 {
            src_lane = wave.threads[lane as usize].read_register(rs_lane);
            break;
        }
    }

    let value = if src_lane < wave_width {
        wave.threads[src_lane as usize].read_register(rs1)
    } else {
        0
    };

    for lane in 0..wave_width {
        if (active_mask & (1u64 << lane)) != 0 {
            wave.threads[lane as usize].write_register(rd, value);
        }
    }
}

pub fn wave_ballot(wave: &mut Wave, rd: u8, pd_src: u8) {
    let wave_width = wave.wave_width;
    let active_mask = wave.active_mask;

    let mut ballot: u64 = 0;
    for lane in 0..wave_width {
        if (active_mask & (1u64 << lane)) != 0 && wave.threads[lane as usize].read_predicate(pd_src)
        {
            ballot |= 1u64 << lane;
        }
    }

    let [a, b, c, d, ..] = ballot.to_le_bytes();
    let ballot_lo = u32::from_le_bytes([a, b, c, d]);

    for lane in 0..wave_width {
        if (active_mask & (1u64 << lane)) != 0 {
            wave.threads[lane as usize].write_register(rd, ballot_lo);
        }
    }
}

pub fn wave_any(wave: &mut Wave, pd_dst: u8, pd_src: u8) {
    let wave_width = wave.wave_width;
    let active_mask = wave.active_mask;

    let mut any_true = false;
    for lane in 0..wave_width {
        if (active_mask & (1u64 << lane)) != 0 && wave.threads[lane as usize].read_predicate(pd_src)
        {
            any_true = true;
            break;
        }
    }

    for lane in 0..wave_width {
        if (active_mask & (1u64 << lane)) != 0 {
            wave.threads[lane as usize].write_predicate(pd_dst, any_true);
        }
    }
}

pub fn wave_all(wave: &mut Wave, pd_dst: u8, pd_src: u8) {
    let wave_width = wave.wave_width;
    let active_mask = wave.active_mask;

    let mut all_true = true;
    for lane in 0..wave_width {
        if (active_mask & (1u64 << lane)) != 0
            && !wave.threads[lane as usize].read_predicate(pd_src)
        {
            all_true = false;
            break;
        }
    }

    for lane in 0..wave_width {
        if (active_mask & (1u64 << lane)) != 0 {
            wave.threads[lane as usize].write_predicate(pd_dst, all_true);
        }
    }
}

pub fn wave_prefix_sum(wave: &mut Wave, rd: u8, rs1: u8) {
    let wave_width = wave.wave_width;
    let active_mask = wave.active_mask;

    let mut prefix = 0u32;
    for lane in 0..wave_width {
        if (active_mask & (1u64 << lane)) != 0 {
            let value = wave.threads[lane as usize].read_register(rs1);
            wave.threads[lane as usize].write_register(rd, prefix);
            prefix = prefix.wrapping_add(value);
        }
    }
}

pub fn wave_reduce_add(wave: &mut Wave, rd: u8, rs1: u8) {
    let wave_width = wave.wave_width;
    let active_mask = wave.active_mask;

    let mut sum = 0u32;
    for lane in 0..wave_width {
        if (active_mask & (1u64 << lane)) != 0 {
            sum = sum.wrapping_add(wave.threads[lane as usize].read_register(rs1));
        }
    }

    for lane in 0..wave_width {
        if (active_mask & (1u64 << lane)) != 0 {
            wave.threads[lane as usize].write_register(rd, sum);
        }
    }
}

pub fn wave_reduce_min(wave: &mut Wave, rd: u8, rs1: u8) {
    let wave_width = wave.wave_width;
    let active_mask = wave.active_mask;

    let mut min_val = u32::MAX;
    for lane in 0..wave_width {
        if (active_mask & (1u64 << lane)) != 0 {
            min_val = min_val.min(wave.threads[lane as usize].read_register(rs1));
        }
    }

    for lane in 0..wave_width {
        if (active_mask & (1u64 << lane)) != 0 {
            wave.threads[lane as usize].write_register(rd, min_val);
        }
    }
}

pub fn wave_reduce_max(wave: &mut Wave, rd: u8, rs1: u8) {
    let wave_width = wave.wave_width;
    let active_mask = wave.active_mask;

    let mut max_val = 0u32;
    for lane in 0..wave_width {
        if (active_mask & (1u64 << lane)) != 0 {
            max_val = max_val.max(wave.threads[lane as usize].read_register(rs1));
        }
    }

    for lane in 0..wave_width {
        if (active_mask & (1u64 << lane)) != 0 {
            wave.threads[lane as usize].write_register(rd, max_val);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_wave() -> Wave {
        let mut wave = Wave::new(crate::wave::WaveConfig {
            wave_width: 8,
            register_count: 32,
            wave_id: 0,
            workgroup_id: [0, 0, 0],
            workgroup_size: [8, 1, 1],
            grid_size: [1, 1, 1],
            base_thread_index: 0,
            total_threads_in_workgroup: 8,
            num_waves: 1,
        });
        for i in 0..8 {
            wave.threads[i].write_register(0, u32::try_from(i * 10).unwrap());
            wave.threads[i].write_register(1, u32::try_from(i).unwrap());
        }
        wave
    }

    #[test]
    fn test_wave_shuffle() {
        let mut wave = create_test_wave();

        for i in 0..8 {
            wave.threads[i].write_register(2, u32::try_from(7 - i).unwrap());
        }

        wave_shuffle(&mut wave, 3, 0, 2);

        assert_eq!(wave.threads[0].read_register(3), 70);
        assert_eq!(wave.threads[7].read_register(3), 0);
    }

    #[test]
    fn test_wave_shuffle_up() {
        let mut wave = create_test_wave();

        for i in 0..8 {
            wave.threads[i].write_register(2, 1);
        }

        wave_shuffle_up(&mut wave, 3, 0, 2);

        assert_eq!(wave.threads[0].read_register(3), 0);
        assert_eq!(wave.threads[1].read_register(3), 0);
        assert_eq!(wave.threads[2].read_register(3), 10);
    }

    #[test]
    fn test_wave_broadcast() {
        let mut wave = create_test_wave();

        for i in 0..8 {
            wave.threads[i].write_register(2, 3);
        }

        wave_broadcast(&mut wave, 3, 0, 2);

        for i in 0..8 {
            assert_eq!(wave.threads[i].read_register(3), 30);
        }
    }

    #[test]
    fn test_wave_ballot() {
        let mut wave = create_test_wave();

        for i in 0..8 {
            wave.threads[i].write_predicate(0, i % 2 == 0);
        }

        wave_ballot(&mut wave, 5, 0);

        for i in 0..8 {
            assert_eq!(wave.threads[i].read_register(5), 0b0101_0101);
        }
    }

    #[test]
    fn test_wave_any() {
        let mut wave = create_test_wave();

        for i in 0..8 {
            wave.threads[i].write_predicate(0, false);
        }
        wave.threads[3].write_predicate(0, true);

        wave_any(&mut wave, 1, 0);

        for i in 0..8 {
            assert!(wave.threads[i].read_predicate(1));
        }
    }

    #[test]
    fn test_wave_all_true() {
        let mut wave = create_test_wave();

        for i in 0..8 {
            wave.threads[i].write_predicate(0, true);
        }

        wave_all(&mut wave, 1, 0);

        for i in 0..8 {
            assert!(wave.threads[i].read_predicate(1));
        }
    }

    #[test]
    fn test_wave_all_false() {
        let mut wave = create_test_wave();

        for i in 0..8 {
            wave.threads[i].write_predicate(0, true);
        }
        wave.threads[5].write_predicate(0, false);

        wave_all(&mut wave, 1, 0);

        for i in 0..8 {
            assert!(!wave.threads[i].read_predicate(1));
        }
    }

    #[test]
    fn test_wave_prefix_sum() {
        let mut wave = create_test_wave();

        for i in 0..8 {
            wave.threads[i].write_register(0, 1);
        }

        wave_prefix_sum(&mut wave, 1, 0);

        for i in 0..8 {
            assert_eq!(wave.threads[i].read_register(1), u32::try_from(i).unwrap());
        }
    }

    #[test]
    fn test_wave_reduce_add() {
        let mut wave = create_test_wave();

        wave_reduce_add(&mut wave, 5, 1);

        for i in 0..8 {
            assert_eq!(wave.threads[i].read_register(5), 28);
        }
    }

    #[test]
    fn test_wave_reduce_min() {
        let mut wave = create_test_wave();

        for i in 0..8 {
            wave.threads[i].write_register(0, u32::try_from(100 - i * 5).unwrap());
        }

        wave_reduce_min(&mut wave, 5, 0);

        for i in 0..8 {
            assert_eq!(wave.threads[i].read_register(5), 65);
        }
    }

    #[test]
    fn test_wave_reduce_max() {
        let mut wave = create_test_wave();

        wave_reduce_max(&mut wave, 5, 0);

        for i in 0..8 {
            assert_eq!(wave.threads[i].read_register(5), 70);
        }
    }

    #[test]
    fn test_wave_ops_respect_active_mask() {
        let mut wave = create_test_wave();
        wave.active_mask = 0b0000_1111;

        for i in 0..8 {
            wave.threads[i].write_register(0, 1);
        }

        wave_reduce_add(&mut wave, 5, 0);

        assert_eq!(wave.threads[0].read_register(5), 4);
        assert_eq!(wave.threads[4].read_register(5), 0);
    }
    #[test]
    fn shuffle_reads_sources_before_overlapping_writes() {
        for width in [1, 4, 32, 64] {
            let mut wave = Wave::new(crate::wave::WaveConfig {
                wave_width: width,
                register_count: 4,
                wave_id: 0,
                workgroup_id: [0; 3],
                workgroup_size: [width, 1, 1],
                grid_size: [1; 3],
                base_thread_index: 0,
                total_threads_in_workgroup: width,
                num_waves: 1,
            });
            for (lane, thread) in wave.threads.iter_mut().enumerate() {
                thread.write_register(0, u32::try_from(lane).unwrap());
                thread.write_register(1, width - 1 - u32::try_from(lane).unwrap());
            }
            wave_shuffle(&mut wave, 0, 0, 1);
            for (lane, thread) in wave.threads.iter().enumerate() {
                assert_eq!(
                    thread.read_register(0),
                    width - 1 - u32::try_from(lane).unwrap()
                );
            }
            wave.active_mask = 1;
            wave_shuffle(&mut wave, 1, 0, 1);
            assert_eq!(wave.threads[0].read_register(1), 0);
            if width > 1 {
                assert_eq!(wave.threads[1].read_register(1), width - 2);
            }
        }
    }

    #[test]
    fn shuffle_down_handles_large_offsets() {
        let mut wave = create_test_wave();
        for thread in &mut wave.threads {
            thread.write_register(2, u32::MAX);
        }
        wave_shuffle_down(&mut wave, 0, 0, 2);
        assert!(wave
            .threads
            .iter()
            .all(|thread| thread.read_register(0) == 0));
    }
}
